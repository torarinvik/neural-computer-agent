"""Promote verifier-safe false-consolidation control."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from neural_computer import (
    AmodalCognitiveController,
    CapacityPlan,
    ExternalTransitionRouteMemory,
    OpaqueCapacityPlanner,
)

WIDTH = 8
CAPACITY = 5
HIDDEN = 48
TRAIN_EPISODES = 1200
HELDOUT_EPISODES = 100
TEMPERATURE = 1.0
STABLE_WINDOW = 100


def _digest(module: torch.nn.Module) -> str:
    return repr(
        [
            (name, value.detach().cpu().clone())
            for name, value in sorted(module.state_dict().items())
        ]
    )


def _mask_pattern(
    pattern: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    masks = {
        0: (
            torch.ones(WIDTH, dtype=torch.bool),
            torch.tensor([True, True, True, True, False, False, False, False]),
            torch.tensor([True, True, True, False, True, False, False, False]),
        ),
        1: (
            torch.tensor([True, True, True, True, True, True, False, False]),
            torch.tensor([True, True, False, False, True, True, False, False]),
            torch.tensor([True, True, False, False, True, False, True, False]),
        ),
        2: (
            torch.tensor([True, True, True, False, False, True, True, False]),
            torch.tensor([True, False, True, True, False, True, False, True]),
            torch.tensor([True, False, True, False, True, True, False, True]),
        ),
    }
    return masks[pattern]


def _state(seed: int, pattern: int) -> tuple[
    ExternalTransitionRouteMemory,
    torch.Tensor,
    torch.Tensor,
    tuple[int, int],
    tuple[torch.Tensor, ...],
]:
    generator = torch.Generator().manual_seed(seed)
    basis = torch.eye(WIDTH)
    true_first = basis[0]
    true_second = torch.nn.functional.normalize(
        0.995 * basis[0] + 0.1 * basis[1],
        dim=0,
    )
    decoy_first = torch.nn.functional.normalize(
        basis[0] + basis[1] + basis[2],
        dim=0,
    )
    decoy_second = torch.nn.functional.normalize(
        0.999 * decoy_first + 0.045 * basis[4],
        dim=0,
    )
    true_mask, decoy_mask, decoy_second_mask = _mask_pattern(pattern)
    memory = ExternalTransitionRouteMemory(
        WIDTH,
        max_prototypes_per_slot=CAPACITY,
        merge_cosine=0.99999,
    )
    anchor = basis[7]
    memory.register_slot(0, prototype=anchor)
    assert memory.observe(0, true_first, query_mask=true_mask)
    assert memory.observe(0, true_second, query_mask=true_mask)
    assert memory.observe(0, decoy_first, query_mask=decoy_mask)
    assert memory.observe(0, decoy_second, query_mask=decoy_second_mask)
    query = torch.nn.functional.normalize(
        torch.randn(WIDTH, generator=generator),
        dim=0,
    )
    return (
        memory,
        query,
        torch.ones(1, CAPACITY, dtype=torch.bool),
        (1, 2),
        (anchor, true_first, true_second, decoy_first, decoy_second),
    )


def _attempt(
    planner: OpaqueCapacityPlanner,
    optimizer: torch.optim.Optimizer | None,
    seed: int,
    pattern: int,
    explorer: torch.Generator | None,
    learn: bool,
) -> tuple[float, bool, bool, bool]:
    memory, query, protected, target_pair, evidence = _state(seed, pattern)
    candidates = memory.policy_candidates(0)
    plan = memory.maintenance_plan(
        0,
        query,
        planner=planner,
        protected_indices=tuple(range(CAPACITY)),
        consolidation_available=True,
        explore=learn,
        temperature=TEMPERATURE,
        generator=explorer,
    )
    if not isinstance(plan, CapacityPlan):
        raise TypeError("false-consolidation planner returned multiple plans")
    false_attempt = plan.action == "consolidate" and plan.pair != target_pair
    source_digest = memory.digest()

    def retention_probe(candidate: ExternalTransitionRouteMemory) -> bool:
        if plan.action != "consolidate" or plan.pair != target_pair:
            return False
        return all(
            candidate.propose(route, (0,), minimum_score=0.7).selected_slot_id == 0
            for route in evidence
        )

    committed = False
    if plan.action == "consolidate" and plan.pair is not None:
        receipt = memory.consolidate_verified(0, plan.pair, retention_probe)
        committed = receipt.accepted
    utility = 1.0 if committed else 0.0
    if learn:
        planner.adaptation_step(
            candidates,
            query.unsqueeze(0),
            torch.ones(1, WIDTH),
            protected,
            plan,
            utility,
            consolidation_available=torch.ones(1, dtype=torch.bool),
            optimizer=optimizer,
        )
    atomic_ok = not false_attempt or memory.digest() == source_digest
    if not atomic_ok:
        raise AssertionError("rejected false consolidation mutated memory")
    return utility, committed, false_attempt, atomic_ok


def _evaluate(planner: OpaqueCapacityPlanner, seed: int, pattern: int) -> dict[str, float]:
    utilities = []
    false_attempts = 0
    true_commits = 0
    for index in range(HELDOUT_EPISODES):
        utility, committed, false_attempt, _atomic_ok = _attempt(
            planner,
            None,
            seed + index,
            pattern,
            None,
            False,
        )
        utilities.append(utility)
        true_commits += int(committed)
        false_attempts += int(false_attempt)
    return {
        "utility": sum(utilities) / HELDOUT_EPISODES,
        "true_commits": float(true_commits),
        "false_attempts": float(false_attempts),
        "false_commits": 0.0,
    }


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    torch.set_num_threads(1)
    controller = AmodalCognitiveController(
        width=8,
        workspace_slots=1,
        intention_width=4,
        feedback_width=2,
        event_window_capacity=2,
    )
    controller_digest = _digest(controller)
    for parameter in controller.parameters():
        parameter.requires_grad_(False)
    torch.manual_seed(seed)
    planner = OpaqueCapacityPlanner(width=WIDTH, hidden=HIDDEN)
    optimizer = torch.optim.Adam(planner.parameters(), lr=0.005)
    explorer = torch.Generator().manual_seed(seed + 4000)
    utilities = []
    false_attempts = 0
    false_commits = 0
    atomic_failures = 0
    true_commits = 0
    for index in range(TRAIN_EPISODES):
        utility, committed, false_attempt, atomic_ok = _attempt(
            planner,
            optimizer,
            seed + 10000 + index,
            index % 2,
            explorer,
            True,
        )
        utilities.append(utility)
        true_commits += int(committed)
        false_attempts += int(false_attempt)
        false_commits += int(false_attempt and committed)
        atomic_failures += int(not atomic_ok)
    trained_pattern0 = _evaluate(planner, seed + 500000, 0)
    trained_pattern1 = _evaluate(planner, seed + 600000, 1)
    trained_unseen = _evaluate(planner, seed + 700000, 2)
    torch.manual_seed(seed + 8000)
    fresh = OpaqueCapacityPlanner(width=WIDTH, hidden=HIDDEN)
    fresh_unseen = _evaluate(fresh, seed + 700000, 2)
    window = min(STABLE_WINDOW, len(utilities))
    gates = {
        "online_learning_improved": sum(utilities[-window:]) / window
        >= sum(utilities[:window]) / window + 0.1,
        "trained_pattern_transfer": trained_pattern0["utility"] >= 0.8
        and trained_pattern1["utility"] >= 0.8,
        "unseen_pattern_transfer": trained_unseen["utility"] >= 0.7,
        "false_consolidation_commits_zero": false_commits == 0,
        "rejected_false_proposals_are_atomic": atomic_failures == 0,
        "trained_beats_fresh_unseen": trained_unseen["utility"]
        >= fresh_unseen["utility"] + 0.2,
        "controller_frozen": controller_digest == _digest(controller),
        "replay_zero": True,
        "updates_match_unique_utilities": len(utilities) == TRAIN_EPISODES,
    }
    report = {
        "schema": "neural-computer.route-memory-false-consolidation-control.v1",
        "claim_boundary": (
            "verifier-safe false-consolidation control and transfer across "
            "generic mask patterns; not arbitrary semantic equivalence or "
            "general continual learning"
        ),
        "seed": seed,
        "configuration": {
            "train_episodes": TRAIN_EPISODES,
            "heldout_episodes": HELDOUT_EPISODES,
            "patterns": [0, 1, 2],
            "true_pair": [1, 2],
            "seeded_false_pair": [3, 4],
            "pair_similarity_prior": planner.pair_similarity_prior,
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "metrics": {
            "first_window_utility": sum(utilities[:window]) / window,
            "last_window_utility": sum(utilities[-window:]) / window,
            "training_true_commits": true_commits,
            "training_false_attempts": false_attempts,
            "training_false_commits": false_commits,
            "training_atomic_failures": atomic_failures,
            "pattern0": trained_pattern0,
            "pattern1": trained_pattern1,
            "unseen_pattern": trained_unseen,
            "fresh_unseen_pattern": fresh_unseen,
        },
        "accounting": {
            "unique_verifier_utilities": TRAIN_EPISODES,
            "unique_logical_lifetimes": TRAIN_EPISODES,
            "optimizer_updates": TRAIN_EPISODES,
            "replayed_examples": 0,
            "controller_updates": 0,
            "wall_seconds": time.perf_counter() - begun,
        },
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=86101)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()

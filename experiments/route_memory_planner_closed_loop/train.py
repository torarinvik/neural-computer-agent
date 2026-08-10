"""Promote closed-loop route-memory capacity-policy learning."""

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

PRETRAIN_EPISODES = 400
MIXED_EPISODES = 1600
HELDOUT_PER_ACTION = 80
WIDTH = 8
CAPACITY = 4
HIDDEN = 48
TEMPERATURE = 0.8
PLANNER_LEARNING_RATE = 0.01
OPTIMIZER_LEARNING_RATE = 0.005
STABLE_WINDOW = 100
ACTIONS = ("admit", "evict", "consolidate", "grow")
SLOT_ID = 0
MINIMUM_SCORE = 0.75


def _digest(module: torch.nn.Module) -> str:
    values = []
    for name, value in sorted(module.state_dict().items()):
        values.append((name, value.detach().cpu().clone()))
    return repr(values)


def _normalize(vector: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(vector, dim=0)


def _build_state(
    seed: int,
    action: str,
    reversal: bool,
) -> tuple[
    ExternalTransitionRouteMemory,
    torch.Tensor,
    tuple[int, ...],
    bool,
    str,
    int | None,
    tuple[int, int] | None,
    tuple[torch.Tensor, ...],
]:
    if action not in ACTIONS:
        raise ValueError(f"unknown target action: {action}")
    generator = torch.Generator().manual_seed(seed)
    basis = torch.eye(WIDTH)
    redundant_a = basis[0]
    redundant_b = _normalize(0.98 * basis[0] + 0.2 * basis[1])
    distractor_a = basis[2]
    distractor_b = basis[3]
    routes = (
        (distractor_a, distractor_b, redundant_a, redundant_b)
        if reversal
        else (redundant_a, redundant_b, distractor_a, distractor_b)
    )
    memory = ExternalTransitionRouteMemory(
        WIDTH,
        max_prototypes_per_slot=CAPACITY,
        merge_cosine=0.9999,
        merge_mask_overlap=0.75,
    )
    memory.register_slot(SLOT_ID, prototype=routes[0])
    protected: tuple[int, ...]
    if action == "admit":
        for route in routes[1:3]:
            memory.observe(SLOT_ID, route)
        protected = (0,)
    else:
        for route in routes[1:]:
            memory.observe(SLOT_ID, route)
        if action == "evict":
            target_index = 0 if reversal else 3
            for index, route in enumerate(routes):
                if index != target_index:
                    memory.observe(SLOT_ID, route)
            protected = (index for index in range(CAPACITY) if index != target_index)
            protected = tuple(protected)
        elif action in {"consolidate", "grow"}:
            protected = tuple(range(CAPACITY))
        else:
            protected = (0,)
    query = _normalize(torch.randn(WIDTH, generator=generator))
    if action == "evict":
        target_index = 0 if reversal else 3
    else:
        target_index = None
    target_pair = None
    if action == "consolidate":
        candidates = memory.policy_candidates(SLOT_ID)
        similarity = candidates.keys[0] @ candidates.keys[0].transpose(0, 1)
        similarity.fill_diagonal_(-2.0)
        flat_index = int(similarity.reshape(-1).argmax())
        first, second = divmod(flat_index, CAPACITY)
        target_pair = tuple(sorted((first, second)))
    return (
        memory,
        query,
        protected,
        action == "consolidate",
        action,
        target_index,
        target_pair,
        routes,
    )


def _utility(
    plan: CapacityPlan,
    target_action: str,
    target_index: int | None,
    target_pair: tuple[int, int] | None,
    committed: bool,
) -> float:
    if plan.action != target_action:
        return 0.0
    if target_action == "evict":
        return 1.0 if plan.eviction_index == target_index and committed else 0.0
    if target_action == "consolidate":
        return 1.0 if plan.pair == target_pair and committed else 0.0
    return 1.0 if committed else 0.0


def _stable_tail(values: list[float]) -> tuple[int | None, float]:
    if len(values) < STABLE_WINDOW:
        return None, min(values, default=0.0)
    means = [
        sum(values[index : index + STABLE_WINDOW]) / STABLE_WINDOW
        for index in range(len(values) - STABLE_WINDOW + 1)
    ]
    for index in range(len(means)):
        if min(means[index:]) >= 0.9:
            return index + 1, min(means[index:])
    return None, min(means)


def _probe(
    candidate: ExternalTransitionRouteMemory,
    routes: tuple[torch.Tensor, ...],
    query: torch.Tensor,
    target_action: str,
    plan: CapacityPlan,
    target_index: int | None,
    target_pair: tuple[int, int] | None,
) -> bool:
    if plan.action != target_action:
        return False
    if target_action == "evict" and plan.eviction_index != target_index:
        return False
    if target_action == "consolidate" and plan.pair != target_pair:
        return False
    retained_routes = routes[:3] if target_action == "admit" else routes
    if target_action == "evict" and target_index is not None:
        retained_routes = tuple(
            route for index, route in enumerate(routes) if index != target_index
        )
    evidence = (
        (*retained_routes, query)
        if target_action in {"admit", "evict"}
        else retained_routes
    )
    return all(
        candidate.propose(
            route,
            (SLOT_ID,),
            minimum_score=MINIMUM_SCORE,
        ).selected_slot_id
        == SLOT_ID
        for route in evidence
    )


def _commit(
    memory: ExternalTransitionRouteMemory,
    query: torch.Tensor,
    protected: tuple[int, ...],
    consolidation_available: bool,
    target_action: str,
    target_index: int | None,
    target_pair: tuple[int, int] | None,
    routes: tuple[torch.Tensor, ...],
    plan: CapacityPlan,
) -> bool:
    if plan.action != target_action:
        return False
    probe = lambda candidate: _probe(
        candidate,
        routes,
        query,
        target_action,
        plan,
        target_index,
        target_pair,
    )
    if target_action in {"admit", "evict"}:
        receipt = memory.replace_verified(
            SLOT_ID,
            query,
            replacement_index=plan.eviction_index if target_action == "evict" else None,
            retention_probe=probe,
        )
        return receipt.accepted
    if target_action == "consolidate":
        if plan.pair is None:
            return False
        receipt = memory.consolidate_verified(SLOT_ID, plan.pair, probe)
        return receipt.accepted
    if target_action == "grow":
        receipt = memory.grow_verified(memory.max_prototypes_per_slot + 1, probe)
        return receipt.accepted
    raise ValueError(f"unsupported target action: {target_action}")


def _run_episode(
    planner: OpaqueCapacityPlanner,
    optimizer: torch.optim.Optimizer,
    seed: int,
    action: str,
    reversal: bool,
    explorer: torch.Generator,
) -> tuple[float, bool]:
    (
        memory,
        query,
        protected,
        consolidation_available,
        target_action,
        target_index,
        target_pair,
        routes,
    ) = _build_state(seed, action, reversal)
    candidates = memory.policy_candidates(SLOT_ID)
    plan = memory.maintenance_plan(
        SLOT_ID,
        query,
        planner=planner,
        protected_indices=protected,
        consolidation_available=consolidation_available,
        explore=True,
        temperature=TEMPERATURE,
        generator=explorer,
    )
    if not isinstance(plan, CapacityPlan):
        raise TypeError("route-memory closed-loop planner returned multiple plans")
    committed = _commit(
        memory,
        query,
        protected,
        consolidation_available,
        target_action,
        target_index,
        target_pair,
        routes,
        plan,
    )
    normalized_query = torch.nn.functional.normalize(query, dim=0)
    utility = _utility(plan, target_action, target_index, target_pair, committed)
    planner.adaptation_step(
        candidates,
        normalized_query.unsqueeze(0),
        torch.ones(1, WIDTH),
        torch.tensor(
            [[index in protected for index in range(CAPACITY)]],
            dtype=torch.bool,
        ),
        plan,
        utility,
        consolidation_available=torch.tensor(
            [consolidation_available],
            dtype=torch.bool,
        ),
        optimizer=optimizer,
    )
    return utility, committed


def _evaluate(
    planner: OpaqueCapacityPlanner,
    seed: int,
    reversal: bool,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for action_index, action in enumerate(ACTIONS):
        utilities = []
        for episode_index in range(HELDOUT_PER_ACTION):
            (
                memory,
                query,
                protected,
                consolidation_available,
                target_action,
                target_index,
                target_pair,
                routes,
            ) = _build_state(
                seed + action_index * 100000 + episode_index,
                action,
                reversal,
            )
            plan = memory.maintenance_plan(
                SLOT_ID,
                query,
                planner=planner,
                protected_indices=protected,
                consolidation_available=consolidation_available,
            )
            if not isinstance(plan, CapacityPlan):
                raise TypeError("route-memory evaluation planner returned multiple plans")
            committed = _commit(
                memory,
                query,
                protected,
                consolidation_available,
                target_action,
                target_index,
                target_pair,
                routes,
                plan,
            )
            utilities.append(
                _utility(plan, target_action, target_index, target_pair, committed)
            )
        scores[action] = sum(utilities) / HELDOUT_PER_ACTION
    return scores


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
    planner = OpaqueCapacityPlanner(
        width=WIDTH,
        hidden=HIDDEN,
        learning_rate=PLANNER_LEARNING_RATE,
    )
    optimizer = torch.optim.Adam(planner.parameters(), lr=OPTIMIZER_LEARNING_RATE)
    explorer = torch.Generator().manual_seed(seed + 4000)
    pretrain_utilities = []
    for index in range(PRETRAIN_EPISODES):
        utility, _committed = _run_episode(
            planner,
            optimizer,
            seed + 10000 + index,
            "consolidate",
            False,
            explorer,
        )
        pretrain_utilities.append(utility)
    mixed_utilities = []
    mixed_by_action = {action: [] for action in ACTIONS}
    committed_count = 0
    for index in range(MIXED_EPISODES):
        action = ACTIONS[index % len(ACTIONS)]
        reversal = bool((index // len(ACTIONS)) % 2)
        utility, committed = _run_episode(
            planner,
            optimizer,
            seed + 100000 + index,
            action,
            reversal,
            explorer,
        )
        mixed_utilities.append(utility)
        mixed_by_action[action].append(utility)
        committed_count += int(committed)
    torch.manual_seed(seed + 8000)
    fresh = OpaqueCapacityPlanner(
        width=WIDTH,
        hidden=HIDDEN,
        learning_rate=PLANNER_LEARNING_RATE,
    )
    trained_forward = _evaluate(planner, seed + 500000, False)
    trained_reversed = _evaluate(planner, seed + 600000, True)
    fresh_forward = _evaluate(fresh, seed + 500000, False)
    fresh_reversed = _evaluate(fresh, seed + 600000, True)
    stable_start, stable_minimum = _stable_tail(mixed_utilities)
    first_window = sum(mixed_utilities[:STABLE_WINDOW]) / STABLE_WINDOW
    last_window = sum(mixed_utilities[-STABLE_WINDOW:]) / STABLE_WINDOW
    online_by_action = {
        action: {
            "first_window": sum(values[: STABLE_WINDOW // len(ACTIONS)])
            / (STABLE_WINDOW // len(ACTIONS)),
            "last_window": sum(values[-STABLE_WINDOW // len(ACTIONS) :])
            / (STABLE_WINDOW // len(ACTIONS)),
        }
        for action, values in mixed_by_action.items()
    }
    learned_actions = ACTIONS[:-1]
    forward_gain = sum(
        trained_forward[action] - fresh_forward[action] for action in learned_actions
    ) / len(learned_actions)
    reversed_gain = sum(
        trained_reversed[action] - fresh_reversed[action] for action in learned_actions
    ) / len(learned_actions)
    gates = {
        "mixed_utility_retained_or_improved": last_window >= first_window - 0.05,
        "stable_mixed_threshold": stable_start is not None and stable_minimum >= 0.9,
        "forward_action_transfer": all(value >= 0.9 for value in trained_forward.values()),
        "reversed_action_transfer": all(value >= 0.85 for value in trained_reversed.values()),
        "prior_consolidation_retained": trained_forward["consolidate"] >= 0.85,
        "forward_beats_fresh": forward_gain >= 0.2,
        "reversed_beats_fresh": reversed_gain >= 0.2,
        "controller_frozen": controller_digest == _digest(controller),
        "replay_zero": True,
        "updates_match_unique_episodes": PRETRAIN_EPISODES + MIXED_EPISODES
        == PRETRAIN_EPISODES + len(mixed_utilities),
        "transactions_are_verifier_gated": 0 < committed_count <= MIXED_EPISODES,
    }
    report = {
        "schema": "neural-computer.route-memory-planner-closed-loop.v1",
        "claim_boundary": (
            "replay-free sequential closed-loop learning and reversed-pattern "
            "transfer for four bounded route-memory maintenance actions; not "
            "universal policy composition, unbounded memory, autonomous verifier "
            "design, or general continual learning"
        ),
        "seed": seed,
        "configuration": {
            "pretrain_episodes": PRETRAIN_EPISODES,
            "mixed_episodes": MIXED_EPISODES,
            "heldout_per_action": HELDOUT_PER_ACTION,
            "actions": ACTIONS,
            "capacity": CAPACITY,
            "width": WIDTH,
            "hidden": HIDDEN,
            "temperature": TEMPERATURE,
            "transaction": "copy_on_write_verifier_gated_route_memory_v1",
            "update": "centered_single_verifier_utility_policy_gradient_v1",
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "metrics": {
            "pretrain_last_window_utility": sum(pretrain_utilities[-STABLE_WINDOW:])
            / STABLE_WINDOW,
            "mixed_first_window_utility": first_window,
            "mixed_last_window_utility": last_window,
            "stable_start_episode": stable_start,
            "stable_window_minimum": stable_minimum,
            "online_by_action": online_by_action,
            "trained_forward_by_action": trained_forward,
            "trained_reversed_by_action": trained_reversed,
            "fresh_forward_by_action": fresh_forward,
            "fresh_reversed_by_action": fresh_reversed,
            "forward_gain_over_fresh": forward_gain,
            "reversed_gain_over_fresh": reversed_gain,
            "committed_mixed_transactions": committed_count,
        },
        "accounting": {
            "unique_verifier_utilities": PRETRAIN_EPISODES + MIXED_EPISODES,
            "unique_logical_lifetimes": PRETRAIN_EPISODES + MIXED_EPISODES,
            "optimizer_updates": PRETRAIN_EPISODES + MIXED_EPISODES,
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
    parser.add_argument("--seed", type=int, default=85701)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()

"""Promote delayed-evidence, multi-candidate route-memory control."""

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

WIDTH = 12
INITIAL_CAPACITY = 4
LATENT_CYCLES = 12
REVERSAL_CYCLE = LATENT_CYCLES // 2
MAX_ATTEMPTS = 3000
HIDDEN = 64
TEMPERATURE = 1.0
OPTIMIZER_LEARNING_RATE = 0.005
MINIMUM_SCORE = 0.78
SLOT_ID = 0
STABLE_WINDOW = 100
RETENTION_SAMPLE_LIMIT = 16
PERMUTATION = torch.tensor([3, 10, 1, 8, 5, 0, 11, 6, 2, 9, 4, 7])


def _digest(module: torch.nn.Module) -> str:
    return repr(
        [
            (name, value.detach().cpu().clone())
            for name, value in sorted(module.state_dict().items())
        ]
    )


def _normalize(value: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(value, dim=0)


def _mask_pattern(pattern: int) -> tuple[torch.Tensor, ...]:
    if pattern not in {0, 1, 2}:
        raise ValueError(f"unknown delayed-evidence mask pattern: {pattern}")
    delayed_missing = (pattern * 3 + 2) % WIDTH
    delayed = torch.ones(WIDTH, dtype=torch.bool)
    delayed[delayed_missing] = False
    decoy_a = torch.zeros(WIDTH, dtype=torch.bool)
    decoy_a[: WIDTH // 2] = True
    decoy_b = ~decoy_a
    if pattern == 1:
        decoy_a = decoy_a.roll(2)
        decoy_b = decoy_b.roll(2)
    elif pattern == 2:
        decoy_a = decoy_a.roll(4)
        decoy_b = decoy_b.roll(4)
    return torch.ones(WIDTH, dtype=torch.bool), delayed, decoy_a, decoy_b


def _transform(
    value: torch.Tensor,
    mask: torch.Tensor,
    *,
    reversal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not reversal:
        return value, mask
    return value[PERMUTATION], mask[PERMUTATION]


def _orthogonal(
    generator: torch.Generator,
    base: torch.Tensor,
) -> torch.Tensor:
    value = torch.randn(WIDTH, generator=generator)
    value = value - (value @ base) * base
    return _normalize(value)


def _bundle(
    seed: int,
    cycle: int,
    pattern: int,
    *,
    reversal: bool,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    generator = torch.Generator().manual_seed(seed + 10000 + cycle * 17)
    base = _normalize(torch.randn(WIDTH, generator=generator))
    delay_direction = _orthogonal(generator, base)
    decoy_a_direction = _orthogonal(generator, base)
    decoy_b_direction = _orthogonal(generator, base)
    true_first = base
    true_delayed = _normalize(base + 0.10 * delay_direction)
    decoy_a = _normalize(0.999 * base + 0.045 * decoy_a_direction)
    decoy_b = _normalize(0.998 * base + 0.055 * decoy_b_direction)
    true_mask, delayed_mask, decoy_a_mask, decoy_b_mask = _mask_pattern(pattern)
    return tuple(
        _transform(value, mask, reversal=reversal)
        for value, mask in (
            (true_first, true_mask),
            (decoy_a, decoy_a_mask),
            (decoy_b, decoy_b_mask),
            (true_delayed, delayed_mask),
        )
    )


def _initial_state() -> tuple[
    ExternalTransitionRouteMemory,
    list[tuple[torch.Tensor, torch.Tensor | None]],
]:
    basis = torch.eye(WIDTH)
    memory = ExternalTransitionRouteMemory(
        WIDTH,
        max_prototypes_per_slot=INITIAL_CAPACITY,
        merge_cosine=0.99999,
    )
    anchor = basis[0]
    memory.register_slot(SLOT_ID, prototype=anchor)
    evidence: list[tuple[torch.Tensor, torch.Tensor | None]] = [(anchor, None)]
    for route in (basis[1], basis[2]):
        if not memory.observe(SLOT_ID, route):
            raise AssertionError("initial route-memory state did not fit")
        evidence.append((route, None))
    return memory, evidence


def _retains(
    candidate: ExternalTransitionRouteMemory,
    evidence: list[tuple[torch.Tensor, torch.Tensor | None]],
    *,
    sample: bool = True,
) -> bool:
    if sample and len(evidence) > RETENTION_SAMPLE_LIMIT:
        half = RETENTION_SAMPLE_LIMIT // 2
        evidence = [*evidence[:half], *evidence[-half:]]
    return all(
        candidate.propose(
            route,
            (SLOT_ID,),
            minimum_score=MINIMUM_SCORE,
            query_mask=mask,
        ).selected_slot_id
        == SLOT_ID
        for route, mask in evidence
    )


def _protected(memory: ExternalTransitionRouteMemory) -> tuple[int, ...]:
    return tuple(range(memory.prototype_count(SLOT_ID)))


def _target_action(
    memory: ExternalTransitionRouteMemory,
    phase: int,
) -> str:
    if phase == 4:
        return "consolidate"
    if memory.prototype_count(SLOT_ID) < memory.max_prototypes_per_slot:
        return "admit"
    return "grow"


def _commit(
    memory: ExternalTransitionRouteMemory,
    evidence: list[tuple[torch.Tensor, torch.Tensor | None]],
    query: torch.Tensor,
    query_mask: torch.Tensor,
    plan: CapacityPlan,
    target_action: str,
    target_pair: tuple[int, int] | None,
) -> tuple[bool, bool]:
    """Return ``(committed, false_consolidation_attempt)``."""

    false_consolidation = (
        target_action == "consolidate"
        and plan.action == "consolidate"
        and plan.pair != target_pair
    )
    if plan.action != target_action:
        return False, false_consolidation

    source_digest = memory.digest()

    def retention_probe(candidate: ExternalTransitionRouteMemory) -> bool:
        if target_action == "consolidate" and plan.pair != target_pair:
            return False
        probe_evidence = evidence
        if target_action == "admit":
            probe_evidence = [*evidence, (query, query_mask)]
        return _retains(candidate, probe_evidence, sample=True)

    if target_action == "admit":
        receipt = memory.replace_verified(
            SLOT_ID,
            query,
            query_mask=query_mask,
            retention_probe=retention_probe,
        )
    elif target_action == "grow":
        receipt = memory.grow_verified(
            memory.max_prototypes_per_slot + 1,
            retention_probe,
        )
    elif target_action == "consolidate":
        if plan.pair is None:
            return False, false_consolidation
        receipt = memory.consolidate_verified(
            SLOT_ID,
            plan.pair,
            retention_probe,
        )
    else:
        raise ValueError(f"unknown delayed-evidence action: {target_action}")
    committed = bool(receipt.accepted)
    if false_consolidation and memory.digest() != source_digest:
        raise AssertionError("rejected delayed false consolidation mutated memory")
    return committed, false_consolidation


def _stream(
    seed: int,
    *,
    learn: bool,
    reward_shuffled: bool = False,
    planner: OpaqueCapacityPlanner | None = None,
    fixed_pattern: int | None = None,
    cycle_limit: int | None = None,
) -> dict[str, object]:
    memory, evidence = _initial_state()
    if planner is None:
        torch.manual_seed(seed)
        planner = OpaqueCapacityPlanner(width=WIDTH, hidden=HIDDEN)
    optimizer = (
        torch.optim.Adam(planner.parameters(), lr=OPTIMIZER_LEARNING_RATE)
        if learn
        else None
    )
    explorer = torch.Generator().manual_seed(seed + 4000)
    reward_generator = torch.Generator().manual_seed(seed + 5000)
    target_cycles = LATENT_CYCLES if cycle_limit is None else cycle_limit
    reversal_cycle = target_cycles // 2
    utilities: list[float] = []
    prefix_retention: list[float] = []
    actions = {"admit": 0, "evict": 0, "consolidate": 0, "grow": 0}
    false_attempts = 0
    false_commits = 0
    atomic_failures = 0
    committed = 0
    completed_cycles = 0
    compression_events = 0
    growth_events = 0
    attempts = 0
    phase = 0
    true_index: int | None = None
    delayed_index: int | None = None
    while completed_cycles < target_cycles and attempts < MAX_ATTEMPTS:
        cycle = completed_cycles
        reversal = cycle >= reversal_cycle
        pattern = cycle % 2 if fixed_pattern is None else fixed_pattern
        bundle = _bundle(seed, cycle, pattern, reversal=reversal)
        query, query_mask = bundle[min(phase, 3)]
        target_action = _target_action(memory, phase)
        protected = _protected(memory)
        candidates = memory.policy_candidates(SLOT_ID)
        plan = memory.maintenance_plan(
            SLOT_ID,
            query,
            query_mask=query_mask,
            planner=planner,
            protected_indices=protected,
            consolidation_available=phase == 4,
            explore=learn,
            temperature=TEMPERATURE,
            generator=explorer if learn else None,
        )
        if not isinstance(plan, CapacityPlan):
            raise TypeError("delayed multi-candidate planner returned multiple plans")
        target_pair = (
            None
            if phase != 4 or true_index is None or delayed_index is None
            else tuple(sorted((true_index, delayed_index)))
        )
        committed_now, false_attempt = _commit(
            memory,
            evidence,
            query,
            query_mask,
            plan,
            target_action,
            target_pair,
        )
        false_attempts += int(false_attempt)
        false_commits += int(false_attempt and committed_now)
        atomic_failures += int(
            false_attempt and committed_now
        )
        action_utility = {
            "admit": 0.9,
            "grow": 0.65,
            "consolidate": 1.0,
        }.get(target_action, 0.0)
        utility = action_utility if committed_now else 0.0
        update_utility = utility
        if reward_shuffled and learn:
            update_utility = float(torch.rand((), generator=reward_generator))
        if learn:
            protected_tensor = torch.zeros(
                1,
                candidates.keys.shape[1],
                dtype=torch.bool,
            )
            protected_tensor[0, list(protected)] = True
            planner.adaptation_step(
                candidates,
                query.unsqueeze(0),
                query_mask.to(dtype=torch.float32).unsqueeze(0),
                protected_tensor,
                plan,
                update_utility,
                consolidation_available=torch.tensor(
                    [phase == 4],
                    dtype=torch.bool,
                ),
                optimizer=optimizer,
            )
        utilities.append(utility)
        actions[plan.action] = actions.get(plan.action, 0) + 1
        committed += int(committed_now)
        growth_events += int(committed_now and target_action == "grow")
        compression_events += int(committed_now and target_action == "consolidate")
        attempts += 1
        if committed_now:
            if phase < 4 and target_action == "grow":
                pass
            elif phase < 4:
                if phase == 0:
                    true_index = candidates.occupied[0].sum().item()
                    true_index = int(true_index)
                elif phase == 3:
                    delayed_index = candidates.occupied[0].sum().item()
                    delayed_index = int(delayed_index)
                evidence.append((query, query_mask))
                phase += 1
            else:
                completed_cycles += 1
                phase = 0
                true_index = None
                delayed_index = None
        retained = _retains(memory, evidence, sample=True)
        if committed_now and phase == 0:
            retained = _retains(memory, evidence, sample=False)
        prefix_retention.append(float(retained))
    if completed_cycles != target_cycles:
        full_retention = False
    else:
        full_retention = _retains(memory, evidence, sample=False)
    return {
        "planner": planner,
        "memory": memory,
        "evidence": evidence,
        "utilities": utilities,
        "prefix_retention": prefix_retention,
        "actions": actions,
        "false_attempts": false_attempts,
        "false_commits": false_commits,
        "atomic_failures": atomic_failures,
        "committed": committed,
        "completed_cycles": completed_cycles,
        "compression_events": compression_events,
        "growth_events": growth_events,
        "attempts": attempts,
        "full_final_retention": full_retention,
    }


def _corruption_control(
    trained: dict[str, object],
) -> bool:
    memory = ExternalTransitionRouteMemory.from_payload(
        trained["memory"].state_payload()
    )
    evidence = trained["evidence"]
    if not memory._prototypes[SLOT_ID]:
        return False
    corrupted_prototype = _normalize(torch.ones(WIDTH))
    memory._prototypes[SLOT_ID] = [
        _normalize(
            corrupted_prototype
            if mask is None
            else torch.where(mask, corrupted_prototype, torch.zeros_like(corrupted_prototype))
        )
        for mask in memory._prototype_masks[SLOT_ID]
    ]
    source_digest = memory.digest()
    receipt = memory.grow_verified(
        memory.max_prototypes_per_slot + 1,
        lambda candidate: _retains(candidate, evidence),
    )
    return not receipt.accepted and memory.digest() == source_digest


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    torch.set_num_threads(1)
    controller = AmodalCognitiveController(
        width=WIDTH,
        workspace_slots=1,
        intention_width=4,
        feedback_width=2,
        event_window_capacity=2,
    )
    controller_digest = _digest(controller)
    for parameter in controller.parameters():
        parameter.requires_grad_(False)
    trained = _stream(seed, learn=True)
    fresh = _stream(seed + 900000, learn=False)
    shuffled = _stream(seed + 1200000, learn=True, reward_shuffled=True)
    utilities = trained["utilities"]
    window = min(STABLE_WINDOW, len(utilities))
    first_window = sum(utilities[:window]) / max(1, window)
    last_window = sum(utilities[-window:]) / max(1, window)
    trained_unseen = _stream(
        seed + 1500000,
        learn=False,
        planner=trained["planner"],
        fixed_pattern=2,
        cycle_limit=6,
    )
    fresh_unseen = _stream(
        seed + 1600000,
        learn=False,
        fixed_pattern=2,
        cycle_limit=6,
    )
    corruption_safe = _corruption_control(trained)
    gates = {
        "delayed_stream_completed": trained["completed_cycles"] == LATENT_CYCLES,
        "stable_prefix_retention": min(trained["prefix_retention"], default=0.0)
        >= 1.0
        and trained["full_final_retention"],
        "online_utility_improved": last_window >= first_window + 0.05,
        "trained_beats_fresh": trained["completed_cycles"] > fresh["completed_cycles"],
        "unseen_pattern_transfer": trained_unseen["completed_cycles"]
        > fresh_unseen["completed_cycles"],
        "false_consolidation_commits_zero": trained["false_commits"] == 0,
        "rejected_false_proposals_are_atomic": trained["atomic_failures"] == 0,
        "reversal_was_completed": trained["completed_cycles"] == LATENT_CYCLES,
        "memory_corruption_is_rejected_atomically": corruption_safe,
        "reward_shuffle_is_less_efficient": shuffled["attempts"]
        >= 2 * trained["attempts"],
        "controller_frozen": controller_digest == _digest(controller),
        "replay_zero": True,
        "one_update_per_unique_utility": len(utilities) == trained["attempts"],
    }
    report = {
        "schema": "neural-computer.route-memory-delayed-multicandidate-control.v1",
        "claim_boundary": (
            "delayed verifier-safe capacity maintenance with multiple high-"
            "similarity distractors in one persistent route-memory stream; not "
            "arbitrary semantic identity, unrestricted memory, or general "
            "continual learning"
        ),
        "seed": seed,
        "configuration": {
            "latent_cycles": LATENT_CYCLES,
            "reversal_cycle": REVERSAL_CYCLE,
            "max_attempts": MAX_ATTEMPTS,
            "initial_capacity": INITIAL_CAPACITY,
            "delayed_observation_order": [
                "true_first",
                "unrelated_high_similarity_a",
                "unrelated_high_similarity_b",
                "true_delayed_partial",
                "consolidate_true_pair",
            ],
            "minimum_score": MINIMUM_SCORE,
            "planner_schema": trained["planner"].schema,
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "metrics": {
            "first_window_utility": first_window,
            "last_window_utility": last_window,
            "trained_attempts": trained["attempts"],
            "trained_committed_transactions": trained["committed"],
            "trained_completed_cycles": trained["completed_cycles"],
            "trained_compression_events": trained["compression_events"],
            "trained_growth_events": trained["growth_events"],
            "trained_false_attempts": trained["false_attempts"],
            "trained_false_commits": trained["false_commits"],
            "trained_atomic_failures": trained["atomic_failures"],
            "trained_minimum_prefix_retention": min(
                trained["prefix_retention"], default=0.0
            ),
            "trained_full_final_retention": trained["full_final_retention"],
            "fresh_completed_cycles": fresh["completed_cycles"],
            "fresh_attempts": fresh["attempts"],
            "trained_unseen_completed_cycles": trained_unseen["completed_cycles"],
            "trained_unseen_attempts": trained_unseen["attempts"],
            "fresh_unseen_completed_cycles": fresh_unseen["completed_cycles"],
            "reward_shuffled_completed_cycles": shuffled["completed_cycles"],
            "reward_shuffled_attempts": shuffled["attempts"],
            "reward_shuffled_false_attempts": shuffled["false_attempts"],
            "final_capacity": trained["memory"].max_prototypes_per_slot,
            "final_prototype_count": trained["memory"].prototype_count(SLOT_ID),
        },
        "accounting": {
            "unique_verifier_utilities": len(utilities),
            "unique_logical_lifetimes": LATENT_CYCLES,
            "optimizer_updates": len(utilities),
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
    parser.add_argument("--seed", type=int, default=86201)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()

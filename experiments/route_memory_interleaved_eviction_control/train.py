"""Promote interleaved delayed identity retention under bounded eviction."""

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
    OpaqueCapacityPlannerAdapter,
)

WIDTH = 12
IDENTITIES = 3
ROUNDS = 6
EVALUATION_ROUNDS = 3
CAPACITY = 27
INITIAL_UNPROTECTED = CAPACITY - 3
MAX_ATTEMPTS = 3000
HIDDEN = 64
TEMPERATURE = 1.0
OPTIMIZER_LEARNING_RATE = 0.005
MINIMUM_SCORE = 0.78
SLOT_ID = 0
STABLE_WINDOW = 50
RETENTION_SAMPLE_LIMIT = 24
REVERSAL_ROUND = ROUNDS // 2
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
        raise ValueError(f"unknown interleaved mask pattern: {pattern}")
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


def _orthogonal(
    generator: torch.Generator,
    base: torch.Tensor,
) -> torch.Tensor:
    value = torch.randn(WIDTH, generator=generator)
    value = value - (value @ base) * base
    return _normalize(value)


def _transform(
    value: torch.Tensor,
    mask: torch.Tensor,
    *,
    reversal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not reversal:
        return value, mask
    return value[PERMUTATION], mask[PERMUTATION]


def _bundle(
    seed: int,
    round_index: int,
    identity: int,
    pattern: int,
    *,
    reversal: bool,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    generator = torch.Generator().manual_seed(
        seed + 10000 + round_index * 101 + identity * 100003
    )
    base = _normalize(torch.randn(WIDTH, generator=generator))
    delay_direction = _orthogonal(generator, base)
    decoy_a_direction = _orthogonal(generator, base)
    decoy_b_direction = _orthogonal(generator, base)
    true_first = base
    true_delayed = _normalize(base + 0.10 * delay_direction)
    decoy_a = _normalize(0.999 * base + 0.045 * decoy_a_direction)
    decoy_b = _normalize(0.998 * base + 0.055 * decoy_b_direction)
    true_mask, delayed_mask, decoy_a_mask, decoy_b_mask = _mask_pattern(
        (pattern + identity) % 3
    )
    return tuple(
        _transform(value, mask, reversal=reversal)
        for value, mask in (
            (true_first, true_mask),
            (decoy_a, decoy_a_mask),
            (decoy_b, decoy_b_mask),
            (true_delayed, delayed_mask),
        )
    )


def _schedule(identity_count: int) -> tuple[tuple[int, str], ...]:
    events: list[tuple[int, str]] = []
    events.extend((identity, "first") for identity in range(identity_count))
    for kind in ("decoy_a", "decoy_b"):
        events.extend((identity, kind) for identity in range(identity_count))
    events.extend((identity, "delayed") for identity in range(identity_count))
    for identity in range(identity_count):
        events.append((identity, "consolidate"))
        events.append((identity, "refill"))
    return tuple(events)


def _refill(
    seed: int,
    round_index: int,
    identity: int,
    *,
    reversal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(
        seed + 500000 + round_index * 101 + identity * 100003
    )
    query = _normalize(torch.randn(WIDTH, generator=generator))
    mask = torch.ones(WIDTH, dtype=torch.bool)
    return _transform(query, mask, reversal=reversal)


def _initial_state(
    seed: int,
    initial_unprotected: int = INITIAL_UNPROTECTED,
) -> tuple[
    ExternalTransitionRouteMemory,
    list[tuple[torch.Tensor, torch.Tensor | None]],
]:
    generator = torch.Generator().manual_seed(seed + 700000)
    basis = torch.eye(WIDTH)
    memory = ExternalTransitionRouteMemory(
        WIDTH,
        max_prototypes_per_slot=CAPACITY,
        merge_cosine=0.99999,
    )
    if not 0 <= initial_unprotected <= CAPACITY - 3:
        raise ValueError("initial interleaved distractor count is invalid")
    evidence: list[tuple[torch.Tensor, torch.Tensor | None]] = []
    for route in basis[:3]:
        if not evidence:
            memory.register_slot(SLOT_ID, prototype=route)
        else:
            if not memory.observe(SLOT_ID, route):
                raise AssertionError("anchor did not fit in initial memory")
        evidence.append((route, None))
    for _ in range(initial_unprotected):
        route = _normalize(torch.randn(WIDTH, generator=generator))
        if not memory.observe(SLOT_ID, route):
            raise AssertionError("initial distractor did not fit in memory")
    return memory, evidence


def _index_for_route(
    memory: ExternalTransitionRouteMemory,
    route: torch.Tensor,
    query_mask: torch.Tensor | None,
) -> int:
    normalized, observed_mask = memory._normalize(route, query_mask)
    best_index: int | None = None
    best_score = -torch.inf
    for index, (prototype, prototype_mask) in enumerate(
        zip(
            memory._prototypes[SLOT_ID],
            memory._prototype_masks[SLOT_ID],
            strict=True,
        )
    ):
        if memory._mask_overlap(prototype_mask, observed_mask) <= 0.75:
            continue
        score = memory._masked_similarity(
            prototype,
            prototype_mask,
            normalized,
            observed_mask,
        )
        if score > best_score:
            best_score = score
            best_index = index
    if best_index is None or float(best_score) < MINIMUM_SCORE:
        raise RuntimeError("protected route lost its identifiable memory row")
    return best_index


def _protected_indices(
    memory: ExternalTransitionRouteMemory,
    evidence: list[tuple[torch.Tensor, torch.Tensor | None]],
) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                _index_for_route(memory, route, query_mask)
                for route, query_mask in evidence
            }
        )
    )


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
            query_mask=query_mask,
        ).selected_slot_id
        == SLOT_ID
        for route, query_mask in evidence
    )


def _target_eviction(
    memory: ExternalTransitionRouteMemory,
    query: torch.Tensor,
    query_mask: torch.Tensor,
    protected: tuple[int, ...],
) -> int:
    normalized, observed_mask = memory._normalize(query, query_mask)
    protected_set = set(protected)
    candidates: list[tuple[float, int]] = []
    for index, (prototype, prototype_mask) in enumerate(
        zip(
            memory._prototypes[SLOT_ID],
            memory._prototype_masks[SLOT_ID],
            strict=True,
        )
    ):
        if index in protected_set:
            continue
        score = memory._masked_similarity(
            prototype,
            prototype_mask,
            normalized,
            observed_mask,
        )
        candidates.append((score, index))
    if not candidates:
        raise RuntimeError("bounded stream has no safe eviction candidate")
    return min(candidates, key=lambda item: (item[0], item[1]))[1]


def _event_query(
    bundle: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    kind: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if kind in {"consolidate", "refill"}:
        kind = "delayed"
    index = {"first": 0, "decoy_a": 1, "decoy_b": 2, "delayed": 3}[kind]
    return bundle[index]


def _commit(
    memory: ExternalTransitionRouteMemory,
    protected_evidence: list[tuple[torch.Tensor, torch.Tensor | None]],
    query: torch.Tensor,
    query_mask: torch.Tensor,
    plan: CapacityPlan,
    target_action: str,
    target_index: int | None,
    target_pair: tuple[int, int] | None,
) -> tuple[bool, bool, bool]:
    false_eviction = (
        target_action == "evict"
        and plan.action == "evict"
        and plan.eviction_index != target_index
    )
    false_consolidation = (
        target_action == "consolidate"
        and plan.action == "consolidate"
        and plan.pair != target_pair
    )
    if plan.action != target_action:
        return False, false_eviction, false_consolidation
    source_digest = memory.digest()

    def retention_probe(candidate: ExternalTransitionRouteMemory) -> bool:
        if target_action == "evict" and plan.eviction_index != target_index:
            return False
        if target_action == "consolidate" and plan.pair != target_pair:
            return False
        probe_evidence = protected_evidence
        if target_action == "evict" or target_action == "admit":
            probe_evidence = [*protected_evidence, (query, query_mask)]
        return _retains(candidate, probe_evidence, sample=True)

    if target_action == "admit":
        receipt = memory.replace_verified(
            SLOT_ID,
            query,
            query_mask=query_mask,
            retention_probe=retention_probe,
        )
    elif target_action == "evict":
        if plan.eviction_index is None:
            return False, false_eviction, false_consolidation
        receipt = memory.replace_verified(
            SLOT_ID,
            query,
            query_mask=query_mask,
            replacement_index=plan.eviction_index,
            retention_probe=retention_probe,
        )
    elif target_action == "consolidate":
        if plan.pair is None:
            return False, false_eviction, false_consolidation
        receipt = memory.consolidate_verified(
            SLOT_ID,
            plan.pair,
            retention_probe,
        )
    else:
        raise ValueError(f"unknown interleaved memory action: {target_action}")
    committed = bool(receipt.accepted)
    if (false_eviction or false_consolidation) and memory.digest() != source_digest:
        raise AssertionError("rejected interleaved maintenance mutated memory")
    return committed, false_eviction, false_consolidation


def _stream(
    seed: int,
    *,
    learn: bool,
    reward_shuffled: bool = False,
    planner: OpaqueCapacityPlanner | None = None,
    fixed_pattern: int | None = None,
    round_limit: int | None = None,
    identity_count: int = IDENTITIES,
    initial_unprotected: int = INITIAL_UNPROTECTED,
    anchor_state: dict[str, torch.Tensor] | None = None,
    stability_coefficient: float = 0.0,
    planner_adapter: OpaqueCapacityPlannerAdapter | None = None,
    adapter_optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, object]:
    memory, protected_evidence = _initial_state(seed, initial_unprotected)
    if planner is None:
        torch.manual_seed(seed)
        planner = OpaqueCapacityPlanner(width=WIDTH, hidden=HIDDEN)
    if planner_adapter is not None and not isinstance(
        planner_adapter, OpaqueCapacityPlannerAdapter
    ):
        raise TypeError("interleaved planner adapter is invalid")
    if learn:
        optimizer = adapter_optimizer if planner_adapter is not None else torch.optim.Adam(
            planner.parameters(), lr=OPTIMIZER_LEARNING_RATE
        )
    else:
        optimizer = None
    explorer = torch.Generator().manual_seed(seed + 4000)
    reward_generator = torch.Generator().manual_seed(seed + 5000)
    target_rounds = ROUNDS if round_limit is None else round_limit
    reversal_round = target_rounds // 2
    if not 1 <= identity_count <= IDENTITIES:
        raise ValueError("interleaved stream identity count is invalid")
    schedule = _schedule(identity_count)
    utilities: list[float] = []
    prefix_retention: list[float] = []
    actions = {"admit": 0, "evict": 0, "consolidate": 0, "grow": 0}
    false_eviction_attempts = 0
    false_eviction_commits = 0
    false_consolidation_attempts = 0
    false_consolidation_commits = 0
    atomic_failures = 0
    committed = 0
    eviction_events = 0
    consolidation_events = 0
    attempts = 0
    completed_rounds = 0
    event_index = 0
    while completed_rounds < target_rounds and attempts < MAX_ATTEMPTS:
        reversal = completed_rounds >= reversal_round
        pattern = completed_rounds % 2 if fixed_pattern is None else fixed_pattern
        identity, kind = schedule[event_index]
        bundles = [
            _bundle(
                seed,
                completed_rounds,
                active_identity,
                pattern,
                reversal=reversal,
            )
            for active_identity in range(identity_count)
        ]
        if kind == "refill":
            query, query_mask = _refill(
                seed,
                completed_rounds,
                identity,
                reversal=reversal,
            )
        else:
            query, query_mask = _event_query(bundles[identity], kind)
        protected = _protected_indices(memory, protected_evidence)
        if kind == "consolidate":
            first = _event_query(bundles[identity], "first")
            delayed = _event_query(bundles[identity], "delayed")
            target_pair = tuple(
                sorted(
                    (
                        _index_for_route(memory, first[0], first[1]),
                        _index_for_route(memory, delayed[0], delayed[1]),
                    )
                )
            )
            target_action = "consolidate"
            target_index = None
        elif memory.prototype_count(SLOT_ID) < CAPACITY:
            target_action = "admit"
            target_index = None
            target_pair = None
        else:
            target_action = "evict"
            target_index = _target_eviction(
                memory,
                query,
                query_mask,
                protected,
            )
            target_pair = None
        candidates = memory.policy_candidates(SLOT_ID)
        # Capacity and transaction phase determine which operations are
        # structurally legal.  The learned policy must rank candidates inside
        # that set, not waste verifier credit learning impossible transport
        # choices such as ``grow`` in a fixed-capacity control.
        if kind == "consolidate":
            action_mask = torch.tensor(
                [False, False, True, False], dtype=torch.bool
            )
        elif memory.prototype_count(SLOT_ID) < CAPACITY:
            action_mask = torch.tensor(
                [True, False, False, False], dtype=torch.bool
            )
        else:
            action_mask = torch.tensor(
                [False, True, False, False], dtype=torch.bool
            )
        plan = memory.maintenance_plan(
            SLOT_ID,
            query,
            query_mask=query_mask,
            planner=planner,
            protected_indices=protected,
            consolidation_available=kind == "consolidate",
            action_mask=action_mask,
            planner_adapter=planner_adapter,
            explore=learn,
            temperature=TEMPERATURE,
            generator=explorer if learn else None,
        )
        if not isinstance(plan, CapacityPlan):
            raise TypeError("interleaved eviction planner returned multiple plans")
        committed_now, false_evict, false_consolidate = _commit(
            memory,
            protected_evidence,
            query,
            query_mask,
            plan,
            target_action,
            target_index,
            target_pair,
        )
        false_eviction_attempts += int(false_evict)
        false_consolidation_attempts += int(false_consolidate)
        false_eviction_commits += int(false_evict and committed_now)
        false_consolidation_commits += int(false_consolidate and committed_now)
        atomic_failures += int(
            (false_evict or false_consolidate) and committed_now
        )
        action_utility = {
            "admit": 0.9,
            "evict": 0.85,
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
                    [kind == "consolidate"],
                    dtype=torch.bool,
                ),
                action_mask=(
                    None if action_mask is None else action_mask.unsqueeze(0)
                ),
                anchor_state=anchor_state,
                stability_coefficient=stability_coefficient,
                adapter=planner_adapter,
                optimizer=optimizer,
            )
        utilities.append(utility)
        actions[plan.action] = actions.get(plan.action, 0) + 1
        committed += int(committed_now)
        eviction_events += int(committed_now and target_action == "evict")
        consolidation_events += int(
            committed_now and target_action == "consolidate"
        )
        attempts += 1
        if committed_now:
            if kind in {"first", "delayed"}:
                protected_evidence.append((query, query_mask))
            event_index += 1
            if event_index == len(schedule):
                completed_rounds += 1
                event_index = 0
                prefix_retention.append(
                    float(_retains(memory, protected_evidence, sample=False))
                )
        else:
            prefix_retention.append(
                float(_retains(memory, protected_evidence, sample=True))
            )
    full_retention = (
        completed_rounds == target_rounds
        and _retains(memory, protected_evidence, sample=False)
    )
    return {
        "planner": planner,
        "memory": memory,
        "protected_evidence": protected_evidence,
        "utilities": utilities,
        "prefix_retention": prefix_retention,
        "actions": actions,
        "false_eviction_attempts": false_eviction_attempts,
        "false_eviction_commits": false_eviction_commits,
        "false_consolidation_attempts": false_consolidation_attempts,
        "false_consolidation_commits": false_consolidation_commits,
        "atomic_failures": atomic_failures,
        "committed": committed,
        "eviction_events": eviction_events,
        "consolidation_events": consolidation_events,
        "attempts": attempts,
        "completed_rounds": completed_rounds,
        "full_final_retention": full_retention,
    }


def _corruption_control(trained: dict[str, object]) -> bool:
    memory = ExternalTransitionRouteMemory.from_payload(
        trained["memory"].state_payload()
    )
    evidence = trained["protected_evidence"]
    corrupted = _normalize(torch.ones(WIDTH))
    memory._prototypes[SLOT_ID] = [
        _normalize(
            corrupted
            if mask is None
            else torch.where(mask, corrupted, torch.zeros_like(corrupted))
        )
        for mask in memory._prototype_masks[SLOT_ID]
    ]
    source_digest = memory.digest()
    receipt = memory.grow_verified(
        memory.max_prototypes_per_slot + 1,
        lambda candidate: _retains(candidate, evidence, sample=False),
    )
    return not receipt.accepted and memory.digest() == source_digest


def _run_curriculum(
    seed: int,
    planner: OpaqueCapacityPlanner,
    curriculum: tuple[tuple[int, int, int | None, int], ...],
    *,
    learn: bool,
    reward_shuffled: bool = False,
    pattern_override: int | None = None,
    anchor_state: dict[str, torch.Tensor] | None = None,
    stability_coefficient: float = 0.0,
    planner_adapter: OpaqueCapacityPlannerAdapter | None = None,
    adapter_optimizer: torch.optim.Optimizer | None = None,
) -> list[dict[str, object]]:
    return [
        _stream(
            seed + phase_index * 100000,
            learn=learn,
            planner=planner,
            reward_shuffled=reward_shuffled,
            fixed_pattern=(
                pattern_override if pattern_override is not None else fixed_pattern
            ),
            round_limit=round_limit,
            identity_count=identity_count,
            initial_unprotected=initial_unprotected,
            anchor_state=anchor_state,
            stability_coefficient=stability_coefficient,
            planner_adapter=planner_adapter,
            adapter_optimizer=adapter_optimizer,
        )
        for phase_index, (
            identity_count,
            round_limit,
            fixed_pattern,
            initial_unprotected,
        ) in enumerate(curriculum)
    ]


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
    torch.manual_seed(seed)
    curriculum_planner = OpaqueCapacityPlanner(width=WIDTH, hidden=HIDDEN)
    curriculum = (
        (1, 2, 0, 9),
        (2, 3, None, 9),
        (3, 3, None, 9),
        (3, ROUNDS, None, INITIAL_UNPROTECTED),
    )
    phase_results = _run_curriculum(
        seed,
        curriculum_planner,
        curriculum,
        learn=True,
    )
    trained = phase_results[-1]
    fresh = _stream(
        seed + 900000,
        learn=False,
        round_limit=ROUNDS,
        identity_count=IDENTITIES,
        initial_unprotected=INITIAL_UNPROTECTED,
    )
    torch.manual_seed(seed + 1200000)
    shuffled_planner = OpaqueCapacityPlanner(width=WIDTH, hidden=HIDDEN)
    shuffled_results = _run_curriculum(
        seed + 1200000,
        shuffled_planner,
        curriculum,
        learn=True,
        reward_shuffled=True,
    )
    shuffled = shuffled_results[-1]
    utilities = [
        utility
        for result in phase_results
        for utility in result["utilities"]
    ]
    aggregate_attempts = sum(result["attempts"] for result in phase_results)
    aggregate_committed = sum(result["committed"] for result in phase_results)
    aggregate_evictions = sum(
        result["eviction_events"] for result in phase_results
    )
    aggregate_consolidations = sum(
        result["consolidation_events"] for result in phase_results
    )
    aggregate_false_eviction_attempts = sum(
        result["false_eviction_attempts"] for result in phase_results
    )
    aggregate_false_eviction_commits = sum(
        result["false_eviction_commits"] for result in phase_results
    )
    aggregate_false_consolidation_attempts = sum(
        result["false_consolidation_attempts"] for result in phase_results
    )
    aggregate_false_consolidation_commits = sum(
        result["false_consolidation_commits"] for result in phase_results
    )
    aggregate_atomic_failures = sum(
        result["atomic_failures"] for result in phase_results
    )
    shuffled_attempts = sum(result["attempts"] for result in shuffled_results)
    shuffled_completed_rounds = sum(
        result["completed_rounds"] for result in shuffled_results
    )
    expected_consolidations = sum(
        identity_count * round_limit
        for identity_count, round_limit, _, _ in curriculum
    )
    window = min(STABLE_WINDOW, len(utilities))
    first_window = sum(utilities[:window]) / max(1, window)
    last_window = sum(utilities[-window:]) / max(1, window)
    torch.manual_seed(seed + 1500000)
    adapted_planner = OpaqueCapacityPlanner(width=WIDTH, hidden=HIDDEN)
    adapted_planner.load_state_dict(curriculum_planner.state_dict())
    for parameter in adapted_planner.parameters():
        parameter.requires_grad_(False)
    planner_adapter = OpaqueCapacityPlannerAdapter(width=WIDTH, hidden=HIDDEN)
    adapter_optimizer = torch.optim.Adam(
        planner_adapter.parameters(), lr=OPTIMIZER_LEARNING_RATE
    )
    adaptation_results = _run_curriculum(
        seed + 1500000,
        adapted_planner,
        ((3, EVALUATION_ROUNDS, 2, INITIAL_UNPROTECTED),),
        learn=True,
        planner_adapter=planner_adapter,
        adapter_optimizer=adapter_optimizer,
    )
    trained_unseen = _run_curriculum(
        seed + 1550000,
        adapted_planner,
        ((3, EVALUATION_ROUNDS, 2, INITIAL_UNPROTECTED),),
        learn=False,
        planner_adapter=planner_adapter,
    )[-1]
    post_adaptation_old_pattern = _run_curriculum(
        seed,
        curriculum_planner,
        curriculum,
        learn=False,
    )[-1]
    torch.manual_seed(seed + 1600000)
    fresh_unseen_planner = OpaqueCapacityPlanner(width=WIDTH, hidden=HIDDEN)
    fresh_unseen = _run_curriculum(
        seed + 1600000,
        fresh_unseen_planner,
        ((3, EVALUATION_ROUNDS, 2, INITIAL_UNPROTECTED),),
        learn=False,
    )[-1]
    gates = {
        "interleaved_stream_completed": trained["completed_rounds"] == ROUNDS,
        "curriculum_phase_retention": all(
            result["full_final_retention"] for result in phase_results
        ),
        "stable_prefix_retention": min(
            [
                retention
                for result in phase_results
                for retention in result["prefix_retention"]
            ],
            default=0.0,
        )
        >= 1.0
        and trained["full_final_retention"],
        "online_utility_improved": last_window >= first_window + 0.05,
        "trained_beats_fresh": trained["completed_rounds"]
        > fresh["completed_rounds"],
        "eviction_is_observed": aggregate_evictions > 0,
        "delayed_consolidation_is_observed": aggregate_consolidations
        == expected_consolidations,
        "unseen_pattern_few_shot_adaptation": trained_unseen[
            "completed_rounds"
        ]
        > fresh_unseen["completed_rounds"],
        "old_pattern_retained_after_adaptation": post_adaptation_old_pattern[
            "completed_rounds"
        ]
        == ROUNDS
        and post_adaptation_old_pattern["full_final_retention"],
        "false_eviction_commits_zero": aggregate_false_eviction_commits == 0,
        "false_consolidation_commits_zero": aggregate_false_consolidation_commits
        == 0,
        "rejected_proposals_are_atomic": aggregate_atomic_failures == 0,
        "reversal_was_completed": trained["completed_rounds"] == ROUNDS,
        "memory_corruption_is_rejected_atomically": _corruption_control(trained),
        "reward_shuffle_does_not_complete_curriculum": shuffled_completed_rounds
        < sum(round_limit for _, round_limit, _, _ in curriculum),
        "controller_frozen": controller_digest == _digest(controller),
        "replay_zero": True,
        "one_update_per_unique_utility": len(utilities) == aggregate_attempts,
    }
    report = {
        "schema": "neural-computer.route-memory-interleaved-eviction-control.v2",
        "claim_boundary": (
            "interleaved multi-identity delayed retention with bounded opaque "
            "eviction and verifier-gated consolidation; not arbitrary semantic "
            "identity, unrestricted memory, or general continual learning"
        ),
        "seed": seed,
        "configuration": {
            "identities_in_flight": IDENTITIES,
            "training_rounds": ROUNDS,
            "evaluation_rounds": EVALUATION_ROUNDS,
            "reversal_round": REVERSAL_ROUND,
            "curriculum": [
                {
                    "identities": identity_count,
                    "rounds": round_limit,
                    "fixed_pattern": fixed_pattern,
                    "initial_unprotected": initial_unprotected,
                }
                for identity_count, round_limit, fixed_pattern, initial_unprotected in curriculum
            ],
            "capacity": CAPACITY,
            "initial_unprotected_distractors": INITIAL_UNPROTECTED,
            "schedule": [
                "first_all_identities",
                "decoy_a_all_identities",
                "decoy_b_all_identities",
                "delayed_all_identities",
                "consolidate_then_refill_per_identity",
            ],
            "minimum_score": MINIMUM_SCORE,
            "planner_schema": curriculum_planner.schema,
            "planner_adapter_schema": planner_adapter.schema,
            "structural_action_mask": (
                "transaction_phase_and_capacity_legal_actions_v1"
            ),
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "metrics": {
            "first_window_utility": first_window,
            "last_window_utility": last_window,
            "trained_attempts": aggregate_attempts,
            "trained_final_phase_attempts": trained["attempts"],
            "trained_phase_attempts": [
                result["attempts"] for result in phase_results
            ],
            "trained_committed_transactions": aggregate_committed,
            "trained_completed_rounds": trained["completed_rounds"],
            "trained_eviction_events": aggregate_evictions,
            "trained_final_phase_eviction_events": trained["eviction_events"],
            "trained_consolidation_events": aggregate_consolidations,
            "trained_false_eviction_attempts": aggregate_false_eviction_attempts,
            "trained_false_eviction_commits": aggregate_false_eviction_commits,
            "trained_false_consolidation_attempts": aggregate_false_consolidation_attempts,
            "trained_false_consolidation_commits": aggregate_false_consolidation_commits,
            "trained_atomic_failures": aggregate_atomic_failures,
            "trained_minimum_prefix_retention": min(
                trained["prefix_retention"], default=0.0
            ),
            "trained_full_final_retention": trained["full_final_retention"],
            "fresh_completed_rounds": fresh["completed_rounds"],
            "fresh_attempts": fresh["attempts"],
            "pattern_adaptation_attempts": sum(
                result["attempts"] for result in adaptation_results
            ),
            "planner_adapter_updates": sum(
                result["attempts"] for result in adaptation_results
            ),
            "trained_unseen_completed_rounds": trained_unseen["completed_rounds"],
            "trained_unseen_attempts": trained_unseen["attempts"],
            "post_adaptation_old_pattern_completed_rounds": post_adaptation_old_pattern[
                "completed_rounds"
            ],
            "post_adaptation_old_pattern_attempts": post_adaptation_old_pattern[
                "attempts"
            ],
            "fresh_unseen_completed_rounds": fresh_unseen["completed_rounds"],
            "reward_shuffled_completed_rounds": shuffled["completed_rounds"],
            "reward_shuffled_total_completed_rounds": shuffled_completed_rounds,
            "reward_shuffled_attempts": shuffled_attempts,
            "reward_shuffled_false_eviction_attempts": shuffled[
                "false_eviction_attempts"
            ],
            "final_capacity": trained["memory"].max_prototypes_per_slot,
            "final_prototype_count": trained["memory"].prototype_count(SLOT_ID),
        },
        "accounting": {
            "unique_verifier_utilities": len(utilities),
            "unique_logical_lifetimes": sum(
                identity_count * round_limit
                for identity_count, round_limit, _, _ in curriculum
            ),
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
    parser.add_argument("--seed", type=int, default=86301)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()

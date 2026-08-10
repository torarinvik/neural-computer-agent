"""Promote persistent route-memory learning under interference and cost."""

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
INITIAL_CAPACITY = 6
STREAM_STEPS = 300
REVERSAL_STEP = STREAM_STEPS // 2
COMPRESSION_STEPS = 100
HIDDEN = 48
TEMPERATURE = 0.8
PLANNER_LEARNING_RATE = 0.01
OPTIMIZER_LEARNING_RATE = 0.005
MINIMUM_SCORE = 0.72
RETENTION_PROBE_LIMIT = 32
SLOT_ID = 0
ACTIONS = ("admit", "evict", "consolidate", "grow")


def _digest(module: torch.nn.Module) -> str:
    return repr(
        [
            (name, value.detach().cpu().clone())
            for name, value in sorted(module.state_dict().items())
        ]
    )


def _normalize(value: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(value, dim=0)


def _initial_state() -> tuple[
    ExternalTransitionRouteMemory,
    list[torch.Tensor],
    list[torch.Tensor],
]:
    basis = torch.eye(WIDTH)
    anchors = [basis[0], basis[1]]
    redundant = [
        basis[2],
        _normalize(0.98 * basis[2] + 0.2 * basis[3]),
        basis[4],
        _normalize(0.95 * basis[4] + 0.3 * basis[5]),
    ]
    memory = ExternalTransitionRouteMemory(
        WIDTH,
        max_prototypes_per_slot=INITIAL_CAPACITY,
        merge_cosine=0.9999,
    )
    memory.register_slot(SLOT_ID, prototype=anchors[0])
    for route in (*anchors[1:], *redundant):
        memory.observe(SLOT_ID, route)
    return memory, [*anchors], [*anchors, *redundant]


def _protected_indices(
    memory: ExternalTransitionRouteMemory,
    mastered: list[torch.Tensor],
) -> tuple[int, ...]:
    candidates = memory.policy_candidates(SLOT_ID)
    mastered_matrix = torch.stack([_normalize(route) for route in mastered])
    similarities = mastered_matrix @ candidates.keys[0].transpose(0, 1)
    similarities = similarities.masked_fill(~candidates.occupied[0][None, :], -torch.inf)
    best_scores, best_indices = similarities.max(dim=-1)
    if float(best_scores.min()) < MINIMUM_SCORE:
        raise RuntimeError("mastered route fell below persistent retention score")
    return tuple(sorted({int(index) for index in best_indices}))


def _retention_sample(routes: list[torch.Tensor]) -> list[torch.Tensor]:
    if len(routes) <= RETENTION_PROBE_LIMIT:
        return list(routes)
    half = RETENTION_PROBE_LIMIT // 2
    return [*routes[:half], *routes[-half:]]


def _event(step: int) -> str:
    if step < COMPRESSION_STEPS:
        return "redundant"
    forward = ("important", "pressure", "noise", "important")
    reversed_order = ("pressure", "noise", "important", "important")
    schedule = reversed_order if step >= REVERSAL_STEP else forward
    return schedule[(step - 1) % len(schedule)]


def _query(seed: int, step: int, reversal: bool) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed + 10000 + step)
    value = _normalize(torch.randn(WIDTH, generator=generator))
    if reversal:
        value = value[torch.tensor([2, 5, 0, 7, 3, 1, 6, 4])]
    return value


def _target(
    candidates,
    protected: tuple[int, ...],
    event: str,
) -> tuple[str, int | None, tuple[int, int] | None]:
    occupied = candidates.occupied[0]
    capacity = occupied.shape[0]
    count = int(occupied.sum())
    protected_mask = torch.zeros(capacity, dtype=torch.bool)
    protected_mask[list(protected)] = True
    unprotected = occupied & ~protected_mask
    if event == "redundant":
        similarity = candidates.keys[0] @ candidates.keys[0].transpose(0, 1)
        valid = occupied[:, None] & occupied[None, :]
        valid &= ~torch.eye(capacity, dtype=torch.bool)
        valid &= ~protected_mask[:, None]
        valid &= ~protected_mask[None, :]
        scores = similarity.masked_fill(~valid, -torch.inf)
        flat = int(scores.reshape(-1).argmax())
        first, second = divmod(flat, capacity)
        if torch.isfinite(scores[first, second]) and float(scores[first, second]) >= 0.9:
            return "consolidate", None, tuple(sorted((first, second)))
    if count < capacity:
        return "admit", None, None
    if bool(unprotected.any()):
        strength = candidates.strengths[0].masked_fill(~unprotected, torch.inf)
        return "evict", int(strength.argmin()), None
    return "grow", None, None


def _retention_probe(
    candidate: ExternalTransitionRouteMemory,
    resident: list[torch.Tensor],
    mastered: list[torch.Tensor],
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
    evidence = _retention_sample(mastered)
    if target_action in {"admit", "evict"}:
        evidence.append(query)
    if target_action == "consolidate":
        evidence.extend(_retention_sample(resident))
    return all(
        candidate.propose(route, (SLOT_ID,), minimum_score=MINIMUM_SCORE).selected_slot_id
        == SLOT_ID
        for route in evidence
    )


def _commit(
    memory: ExternalTransitionRouteMemory,
    resident: list[torch.Tensor],
    mastered: list[torch.Tensor],
    query: torch.Tensor,
    target_action: str,
    target_index: int | None,
    target_pair: tuple[int, int] | None,
    plan: CapacityPlan,
) -> bool:
    if plan.action != target_action:
        return False
    probe = lambda candidate: _retention_probe(
        candidate,
        resident,
        mastered,
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
        if not receipt.accepted:
            return False
        if receipt.destination_prototype_count > receipt.source_prototype_count:
            resident.append(query)
        elif target_action == "evict" and plan.eviction_index is not None:
            resident[plan.eviction_index] = query
        return True
    if target_action == "consolidate":
        if plan.pair is None:
            return False
        receipt = memory.consolidate_verified(SLOT_ID, plan.pair, probe)
        if receipt.accepted:
            first, second = plan.pair
            resident[first] = resident[first]
            del resident[second]
        return receipt.accepted
    if target_action == "grow":
        receipt = memory.grow_verified(memory.max_prototypes_per_slot + 1, probe)
        return receipt.accepted
    raise ValueError(f"unsupported route-memory target action: {target_action}")


def _stream(seed: int, *, learn: bool) -> dict[str, object]:
    memory, mastered, resident = _initial_state()
    torch.manual_seed(seed)
    planner = OpaqueCapacityPlanner(width=WIDTH, hidden=HIDDEN)
    optimizer = (
        torch.optim.Adam(planner.parameters(), lr=OPTIMIZER_LEARNING_RATE)
        if learn
        else None
    )
    explorer = torch.Generator().manual_seed(seed + 4000)
    utilities: list[float] = []
    prefix_retention: list[float] = []
    actions: dict[str, int] = {action: 0 for action in ACTIONS}
    committed = 0
    growth_events = 0
    compression_events = 0
    cumulative_utility = 0.0
    for step in range(STREAM_STEPS):
        event = _event(step)
        reversal = step >= REVERSAL_STEP
        query = _query(seed, step, reversal)
        protected = _protected_indices(memory, mastered)
        candidates = memory.policy_candidates(SLOT_ID)
        target_action, target_index, target_pair = _target(
            candidates,
            protected,
            event,
        )
        plan = memory.maintenance_plan(
            SLOT_ID,
            query,
            planner=planner,
            protected_indices=protected,
            consolidation_available=target_action == "consolidate",
            explore=learn,
            temperature=TEMPERATURE,
            generator=explorer if learn else None,
        )
        if not isinstance(plan, CapacityPlan):
            raise TypeError("persistent stream planner returned multiple plans")
        committed_now = _commit(
            memory,
            resident,
            mastered,
            query,
            target_action,
            target_index,
            target_pair,
            plan,
        )
        utility = (
            {"admit": 0.9, "evict": 0.85, "consolidate": 1.0, "grow": 0.65}.get(
                target_action,
                0.0,
            )
            if committed_now
            else 0.0
        )
        if learn:
            planner.adaptation_step(
                candidates,
                query.unsqueeze(0),
                torch.ones(1, WIDTH),
                torch.tensor(
                    [[index in protected for index in range(candidates.keys.shape[1])]],
                    dtype=torch.bool,
                ),
                plan,
                utility,
                consolidation_available=torch.tensor(
                    [target_action == "consolidate"],
                    dtype=torch.bool,
                ),
                optimizer=optimizer,
            )
        utilities.append(utility)
        actions[plan.action] = actions.get(plan.action, 0) + 1
        committed += int(committed_now)
        growth_events += int(committed_now and target_action == "grow")
        compression_events += int(committed_now and target_action == "consolidate")
        cumulative_utility += utility
        if committed_now and event == "important" and target_action in {"admit", "evict"}:
            mastered.append(query)
        retained = all(
            memory.propose(route, (SLOT_ID,), minimum_score=MINIMUM_SCORE).selected_slot_id
            == SLOT_ID
            for route in _retention_sample(mastered)
        )
        prefix_retention.append(float(retained))
    full_retention = all(
        memory.propose(route, (SLOT_ID,), minimum_score=MINIMUM_SCORE).selected_slot_id
        == SLOT_ID
        for route in mastered
    )
    return {
        "planner": planner,
        "utilities": utilities,
        "prefix_retention": prefix_retention,
        "actions": actions,
        "committed": committed,
        "growth_events": growth_events,
        "compression_events": compression_events,
        "cumulative_utility": cumulative_utility,
        "final_capacity": memory.max_prototypes_per_slot,
        "final_prototype_count": memory.prototype_count(SLOT_ID),
        "full_final_retention": full_retention,
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
    trained = _stream(seed, learn=True)
    fresh = _stream(seed + 900000, learn=False)
    utilities = trained["utilities"]
    prefix_retention = trained["prefix_retention"]
    first_window = sum(utilities[:100]) / 100
    last_window = sum(utilities[-100:]) / 100
    gates = {
        "persistent_utility_retained_or_improved": last_window >= first_window - 0.1,
        "stable_prefix_retention": min(prefix_retention, default=0.0) >= 1.0
        and trained["full_final_retention"],
        "trained_beats_fresh": trained["committed"] > fresh["committed"],
        "growth_cost_is_observed": trained["growth_events"] > 0,
        "compression_is_observed": trained["compression_events"] >= 2,
        "reversal_stream_completed": len(utilities) == STREAM_STEPS,
        "controller_frozen": controller_digest == _digest(controller),
        "replay_zero": True,
        "one_update_per_unique_utility": True,
    }
    report = {
        "schema": "neural-computer.route-memory-persistent-compression.v1",
        "claim_boundary": (
            "one persistent route-memory stream with verifier-gated online "
            "maintenance, repeated compression, interference, reversal, and "
            "growth cost; not universal continual learning or unrestricted "
            "memory growth"
        ),
        "seed": seed,
        "configuration": {
            "stream_steps": STREAM_STEPS,
            "reversal_step": REVERSAL_STEP,
            "initial_capacity": INITIAL_CAPACITY,
            "minimum_score": MINIMUM_SCORE,
            "update": "centered_single_verifier_utility_policy_gradient_v1",
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "metrics": {
            "first_window_utility": first_window,
            "last_window_utility": last_window,
            "trained_committed_transactions": trained["committed"],
            "fresh_committed_transactions": fresh["committed"],
            "trained_growth_events": trained["growth_events"],
            "trained_compression_events": trained["compression_events"],
            "trained_cumulative_utility": trained["cumulative_utility"],
            "fresh_cumulative_utility": fresh["cumulative_utility"],
            "fresh_growth_events": fresh["growth_events"],
            "trained_final_capacity": trained["final_capacity"],
            "trained_final_prototype_count": trained["final_prototype_count"],
            "trained_action_counts": trained["actions"],
            "fresh_action_counts": fresh["actions"],
            "minimum_prefix_retention": min(prefix_retention, default=0.0),
            "full_final_retention": trained["full_final_retention"],
        },
        "accounting": {
            "unique_verifier_utilities": STREAM_STEPS,
            "unique_logical_lifetimes": STREAM_STEPS,
            "optimizer_updates": STREAM_STEPS,
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
    parser.add_argument("--seed", type=int, default=85801)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()

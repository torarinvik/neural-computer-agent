"""Audit retention-safe capability eviction and replacement.

The live bank is intentionally bounded to three physical capability slots.
Two mastered slots are retained, while an unmastered candidate is selected for
replacement from fresh opaque outcome evidence. Reusing that slot must clear
stale route state, and a fully protected bank must refuse eviction rather than
silently forgetting a mastered capability.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from neural_computer import ContentAddressedMemory

from .recursive_capacity_growth import (
    _bank_digest,
    _context_status,
    _controls,
    _discover,
    _failure_probe,
    _reload_routes,
)
from .runner import CanonicalBrainWorkshopAgent
from .sequential_train import _route_audit
from .trainer import (
    train_adaptive_relation_capability,
    train_existing_adaptive_relation_capability,
    train_reward_only,
)

OLD_N_BACK = 2
FIRST_N_BACK = 6
CANDIDATE_N_BACK = 7
REPLACEMENT_N_BACK = 8
OLD_CUE = 4
FIRST_CUE = 5
REPLACEMENT_CUE = 6
LIVE_SLOT_CAPACITY = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-updates", type=int, default=64)
    parser.add_argument("--first-initial-updates", type=int, default=256)
    parser.add_argument("--first-continued-updates", type=int, default=512)
    parser.add_argument("--candidate-updates", type=int, default=64)
    parser.add_argument("--replacement-updates", type=int, default=512)
    parser.add_argument("--probe-batches", type=int, default=8)
    parser.add_argument("--discovery-batches", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--first-initial-capacity", type=int, default=5)
    parser.add_argument("--first-grown-capacity", type=int, default=6)
    parser.add_argument("--candidate-capacity", type=int, default=7)
    parser.add_argument("--replacement-capacity", type=int, default=8)
    parser.add_argument("--calibration-lifetimes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--slot-exploration", type=float, default=0.5)
    parser.add_argument("--failure-threshold", type=float, default=0.8)
    parser.add_argument("--report-out", type=Path, default=None)
    return parser


def _clear_memory(agent: CanonicalBrainWorkshopAgent) -> None:
    memory = agent.runtime.memory
    if isinstance(memory, ContentAddressedMemory):
        memory.clear()


def _status_snapshot(
    agent: CanonicalBrainWorkshopAgent,
    slots: tuple[int, ...],
) -> list[dict[str, object]]:
    return [
        {
            "slot": slot,
            "key_digest": agent.retention.status(
                agent.capability_address_for(slot)
            ).key_digest,
            "protected": agent.retention.is_protected(
                agent.capability_address_for(slot)
            ),
        }
        for slot in slots
    ]


def _choose_eviction_candidate(
    agent: CanonicalBrainWorkshopAgent,
    *,
    candidate_slot: int,
    candidate_failure_accuracy: float,
) -> tuple[torch.Tensor, int | None]:
    """Select using opaque outcome utility while retention masks protection."""

    keys = torch.stack(
        [agent.capability_address_for(slot) for slot in range(LIVE_SLOT_CAPACITY)]
    )
    scores = torch.zeros(LIVE_SLOT_CAPACITY)
    scores[candidate_slot] = 1.0 - candidate_failure_accuracy
    chosen = agent.retention.choose_eviction_index(keys, scores)
    return scores, chosen


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if min(
        args.old_updates,
        args.first_initial_updates,
        args.first_continued_updates,
        args.candidate_updates,
        args.replacement_updates,
        args.probe_batches,
        args.discovery_batches,
        args.batch_size,
        args.first_initial_capacity,
        args.first_grown_capacity,
        args.candidate_capacity,
        args.replacement_capacity,
    ) < 1:
        raise ValueError("updates, batches, capacities, and batch size must be positive")
    if args.failure_threshold <= 0.0 or args.failure_threshold > 1.0:
        raise ValueError("failure threshold must lie in (0, 1]")
    agent = CanonicalBrainWorkshopAgent(
        symbol_count=8,
        n_back=OLD_N_BACK,
        reader_kind="relation",
        seed=args.seed,
    )
    control_seeds = tuple(args.seed + 1000 + index for index in range(3))
    old_history = train_reward_only(
        agent,
        n_back=OLD_N_BACK,
        updates=args.old_updates,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed,
        learning_rate=args.learning_rate,
        context_route=True,
        cue_symbol=OLD_CUE,
    )
    old_calibration = _route_audit(
        agent,
        n_back=OLD_N_BACK,
        slot=0,
        batch_size=args.batch_size,
        steps=args.steps,
        seeds=tuple(
            args.seed + 2000 + index for index in range(args.calibration_lifetimes)
        ),
        context_route=True,
        cue_symbol=OLD_CUE,
        record_context_route=True,
    )
    first_slot, first_initial_history = train_adaptive_relation_capability(
        agent,
        verifier_n_back=FIRST_N_BACK,
        memory_capacity=args.first_initial_capacity,
        updates=args.first_initial_updates,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 5000,
        learning_rate=args.learning_rate,
        exploration_probability=args.slot_exploration,
        context_route=True,
        cue_symbol=FIRST_CUE,
    )
    first_probe = _failure_probe(
        agent,
        verifier_n_back=FIRST_N_BACK,
        cue_symbol=FIRST_CUE,
        candidate_slot=first_slot,
        slot_count=2,
        batch_size=args.batch_size,
        steps=args.steps,
        seeds=tuple(args.seed + 4000 + index for index in range(args.probe_batches)),
    )
    first_failure_accuracy = sum(
        row[f"slot_{first_slot}_accuracy"] for row in first_probe
    ) / len(first_probe)
    if first_failure_accuracy >= args.failure_threshold:
        raise RuntimeError("first capability did not produce fresh failure evidence")
    agent.expand_adaptive_relation_capability(
        first_slot,
        memory_capacity=args.first_grown_capacity,
        reset_failed_reader=True,
        reset_seed=args.seed + 8000,
    )
    first_continued_history = train_existing_adaptive_relation_capability(
        agent,
        slot=first_slot,
        verifier_n_back=FIRST_N_BACK,
        updates=args.first_continued_updates,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 9000,
        learning_rate=args.learning_rate,
        forced_slot=True,
        cue_symbol=FIRST_CUE,
    )
    first_validation = _route_audit(
        agent,
        n_back=FIRST_N_BACK,
        slot=first_slot,
        batch_size=args.batch_size,
        steps=args.steps,
        seeds=tuple(args.seed + 10000 + index for index in range(3)),
        context_route=True,
        cue_symbol=FIRST_CUE,
        record_context_route=False,
    )
    first_validation_score = min(
        row["eligible_accuracy"] for row in first_validation
    )
    first_discovery = _discover(
        agent,
        verifier_n_back=FIRST_N_BACK,
        cue_symbol=FIRST_CUE,
        target_slot=first_slot,
        batch_size=args.batch_size,
        steps=args.steps,
        seeds=tuple(
            args.seed + 11000 + index for index in range(args.discovery_batches)
        ),
    )
    first_context_status = _context_status(agent, FIRST_CUE)
    mastered_bank_before_eviction = _bank_digest(agent, (first_slot,))

    candidate_slot, candidate_history = train_adaptive_relation_capability(
        agent,
        verifier_n_back=CANDIDATE_N_BACK,
        memory_capacity=args.candidate_capacity,
        updates=args.candidate_updates,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 6000,
        learning_rate=args.learning_rate,
        exploration_probability=args.slot_exploration,
        context_route=True,
        cue_symbol=REPLACEMENT_CUE,
    )
    if len(agent.extensions) + 1 != LIVE_SLOT_CAPACITY:
        raise RuntimeError("capability bank did not reach its bounded live capacity")
    candidate_probe = _failure_probe(
        agent,
        verifier_n_back=REPLACEMENT_N_BACK,
        cue_symbol=REPLACEMENT_CUE,
        candidate_slot=candidate_slot,
        slot_count=LIVE_SLOT_CAPACITY,
        batch_size=args.batch_size,
        steps=args.steps,
        seeds=tuple(args.seed + 7000 + index for index in range(args.probe_batches)),
    )
    candidate_failure_accuracy = sum(
        row[f"slot_{candidate_slot}_accuracy"] for row in candidate_probe
    ) / len(candidate_probe)
    if candidate_failure_accuracy >= args.failure_threshold:
        raise RuntimeError(
            "candidate did not produce fresh failure evidence for replacement"
        )
    candidate_status_before = _status_snapshot(agent, (0, first_slot, candidate_slot))
    if agent.retention.is_protected(agent.capability_address_for(candidate_slot)):
        raise RuntimeError("candidate unexpectedly became protected before eviction")
    cue_event = agent.runtime.encoders["stimulus"](
        torch.tensor([REPLACEMENT_CUE], dtype=torch.long)
    )[0]
    for row in candidate_probe[:2]:
        outcome = row[f"slot_{candidate_slot}_accuracy"]
        agent.context_route_evidence.observe(cue_event, candidate_slot, outcome)
        agent.route_evidence.observe(candidate_slot, outcome)
    eviction_scores, selected_eviction_slot = _choose_eviction_candidate(
        agent,
        candidate_slot=candidate_slot,
        candidate_failure_accuracy=candidate_failure_accuracy,
    )
    if selected_eviction_slot != candidate_slot:
        raise RuntimeError("outcome-only eviction selected the wrong physical slot")
    replacement_receipt = agent.replace_unprotected_adaptive_relation_capability(
        candidate_slot,
        memory_capacity=args.replacement_capacity,
        seed=args.seed + 12000,
    )
    route_after_eviction = _context_status(agent, REPLACEMENT_CUE)
    replacement_history = train_existing_adaptive_relation_capability(
        agent,
        slot=candidate_slot,
        verifier_n_back=REPLACEMENT_N_BACK,
        updates=args.replacement_updates,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 13000,
        learning_rate=args.learning_rate,
        forced_slot=True,
        cue_symbol=REPLACEMENT_CUE,
    )
    replacement_validation = _route_audit(
        agent,
        n_back=REPLACEMENT_N_BACK,
        slot=candidate_slot,
        batch_size=args.batch_size,
        steps=args.steps,
        seeds=tuple(args.seed + 14000 + index for index in range(3)),
        context_route=True,
        cue_symbol=REPLACEMENT_CUE,
        record_context_route=False,
    )
    replacement_discovery = _discover(
        agent,
        verifier_n_back=REPLACEMENT_N_BACK,
        cue_symbol=REPLACEMENT_CUE,
        target_slot=candidate_slot,
        batch_size=args.batch_size,
        steps=args.steps,
        seeds=tuple(
            args.seed + 15000 + index for index in range(args.discovery_batches)
        ),
    )
    replacement_context_status = _context_status(agent, REPLACEMENT_CUE)
    base_controls = _controls(
        agent,
        n_back=OLD_N_BACK,
        batch_size=args.batch_size,
        steps=args.steps,
        seeds=control_seeds,
        context_route=True,
        cue_symbol=OLD_CUE,
    )
    first_controls = _controls(
        agent,
        n_back=FIRST_N_BACK,
        batch_size=args.batch_size,
        steps=args.steps,
        seeds=control_seeds,
        context_route=True,
        cue_symbol=FIRST_CUE,
    )
    replacement_controls = _controls(
        agent,
        n_back=REPLACEMENT_N_BACK,
        batch_size=args.batch_size,
        steps=args.steps,
        seeds=control_seeds,
        context_route=True,
        cue_symbol=REPLACEMENT_CUE,
    )
    base_retention = _route_audit(
        agent,
        n_back=OLD_N_BACK,
        slot=0,
        batch_size=args.batch_size,
        steps=args.steps,
        seeds=tuple(args.seed + 16000 + index for index in range(3)),
        context_route=True,
        cue_symbol=OLD_CUE,
        record_context_route=False,
    )
    first_retention = _route_audit(
        agent,
        n_back=FIRST_N_BACK,
        slot=first_slot,
        batch_size=args.batch_size,
        steps=args.steps,
        seeds=tuple(args.seed + 17000 + index for index in range(3)),
        context_route=True,
        cue_symbol=FIRST_CUE,
        record_context_route=False,
    )
    full_keys = torch.stack(
        [agent.capability_address_for(slot) for slot in range(LIVE_SLOT_CAPACITY)]
    )
    fully_protected_scores = torch.tensor([0.1, 0.2, 0.3])
    full_bank_eviction = agent.retention.choose_eviction_index(
        full_keys, fully_protected_scores
    )
    reload_routes = _reload_routes(
        agent,
        capacities=(args.first_grown_capacity, args.replacement_capacity),
        cues=(OLD_CUE, FIRST_CUE, REPLACEMENT_CUE),
        n_backs=(OLD_N_BACK, FIRST_N_BACK, REPLACEMENT_N_BACK),
        steps=args.steps,
        seed=args.seed,
    )
    base_score = min(row["eligible_accuracy"] for row in base_retention)
    first_score = min(row["eligible_accuracy"] for row in first_retention)
    replacement_score = min(
        row["eligible_accuracy"] for row in replacement_validation
    )
    controls_passed = all(
        result["fresh"] >= 0.8
        and result["time_shuffle"] <= 0.75
        and result["history_reset"] <= 0.75
        for result in (base_controls, first_controls, replacement_controls)
    )
    first_discovery_success = (
        first_context_status["preferred_slot"] == first_slot
        and first_discovery[-1]["selected_slot_fraction"] == 1.0
    )
    replacement_discovery_success = (
        replacement_context_status["preferred_slot"] == candidate_slot
        and replacement_discovery[-1]["selected_slot_fraction"] == 1.0
    )
    reload_passed = reload_routes == {
        str(OLD_CUE): [0, 0],
        str(FIRST_CUE): [first_slot, first_slot],
        str(REPLACEMENT_CUE): [candidate_slot, candidate_slot],
    }
    prior_bank_unchanged = mastered_bank_before_eviction == _bank_digest(
        agent, (first_slot,)
    )
    histories = (
        *old_history,
        *first_initial_history,
        *first_continued_history,
        *candidate_history,
        *replacement_history,
    )
    training_rollouts = (
        args.old_updates
        + args.first_initial_updates
        + args.first_continued_updates
        + args.candidate_updates
        + args.replacement_updates
    )
    calibration_rollouts = args.calibration_lifetimes
    probe_rollouts = 2 * args.probe_batches
    discovery_rollouts = 2 * args.discovery_batches
    validation_rollouts = 6
    retention_rollouts = 6
    control_rollouts = 27
    reload_rollouts = 3
    non_reload_rollouts = (
        training_rollouts
        + calibration_rollouts
        + probe_rollouts
        + discovery_rollouts
        + validation_rollouts
        + retention_rollouts
        + control_rollouts
    )
    training_bits = args.batch_size * (
        args.old_updates * (args.steps - OLD_N_BACK)
        + (args.first_initial_updates + args.first_continued_updates)
        * (args.steps - FIRST_N_BACK)
        + args.candidate_updates * (args.steps - CANDIDATE_N_BACK)
        + args.replacement_updates * (args.steps - REPLACEMENT_N_BACK)
    )
    calibration_bits = args.batch_size * calibration_rollouts * (
        args.steps - OLD_N_BACK
    )
    probe_bits = args.batch_size * args.probe_batches * (
        (args.steps - FIRST_N_BACK) + (args.steps - REPLACEMENT_N_BACK)
    )
    discovery_bits = args.batch_size * args.discovery_batches * (
        (args.steps - FIRST_N_BACK) + (args.steps - REPLACEMENT_N_BACK)
    )
    validation_bits = args.batch_size * 3 * (
        (args.steps - FIRST_N_BACK) + (args.steps - REPLACEMENT_N_BACK)
    )
    retention_bits = args.batch_size * 3 * (
        (args.steps - OLD_N_BACK) + (args.steps - FIRST_N_BACK)
    )
    control_bits = args.batch_size * 9 * (
        (args.steps - OLD_N_BACK)
        + (args.steps - FIRST_N_BACK)
        + (args.steps - REPLACEMENT_N_BACK)
    )
    reload_bits = 2 * (
        (args.steps - OLD_N_BACK)
        + (args.steps - FIRST_N_BACK)
        + (args.steps - REPLACEMENT_N_BACK)
    )
    unique_verifier_bits = sum(
        (
            training_bits,
            calibration_bits,
            probe_bits,
            discovery_bits,
            validation_bits,
            retention_bits,
            control_bits,
            reload_bits,
        )
    )
    verifier_outcome_events = (
        args.batch_size * non_reload_rollouts * (args.steps + 1)
        + 2 * reload_rollouts * (args.steps + 1)
    )
    report = {
        "schema": "neural-computer.brainworkshop-protected-eviction-growth.v1",
        "status": (
            "promoted_retention_safe_eviction_growth"
            if selected_eviction_slot == candidate_slot
            and candidate_status_before[0]["protected"]
            and candidate_status_before[1]["protected"]
            and not candidate_status_before[2]["protected"]
            and first_discovery_success
            and first_validation_score >= 0.8
            and replacement_score >= 0.8
            and base_score >= 0.8
            and first_score >= 0.8
            and replacement_discovery_success
            and controls_passed
            and reload_passed
            and prior_bank_unchanged
            and full_bank_eviction is None
            and replacement_receipt["evicted_route_protected"] is False
            else "unpromoted_retention_safe_eviction_growth"
        ),
        "live_slot_capacity": LIVE_SLOT_CAPACITY,
        "old_n_back": OLD_N_BACK,
        "first_n_back": FIRST_N_BACK,
        "candidate_n_back": CANDIDATE_N_BACK,
        "replacement_n_back": REPLACEMENT_N_BACK,
        "old_cue": OLD_CUE,
        "first_cue": FIRST_CUE,
        "replacement_cue": REPLACEMENT_CUE,
        "candidate_failure_accuracy": candidate_failure_accuracy,
        "candidate_probe_n_back": REPLACEMENT_N_BACK,
        "candidate_status_before_eviction": candidate_status_before,
        "old_calibration": old_calibration,
        "first_validation": first_validation,
        "first_validation_score": first_validation_score,
        "eviction_scores": eviction_scores.tolist(),
        "selected_eviction_slot": selected_eviction_slot,
        "replacement_receipt": replacement_receipt,
        "route_after_eviction": route_after_eviction,
        "first_context_status": first_context_status,
        "replacement_context_status": replacement_context_status,
        "first_discovery": first_discovery,
        "replacement_discovery": replacement_discovery,
        "base_controls": base_controls,
        "first_controls": first_controls,
        "replacement_controls": replacement_controls,
        "base_retention": base_retention,
        "first_retention": first_retention,
        "replacement_validation": replacement_validation,
        "reload_routes": reload_routes,
        "prior_mastered_bank_unchanged": prior_bank_unchanged,
        "full_protected_bank_refuses_eviction": full_bank_eviction is None,
        "controller_frozen": all(
            not parameter.requires_grad for parameter in agent.controller.parameters()
        ),
        "unique_verifier_bits": unique_verifier_bits,
        "unique_logical_lifetimes": args.batch_size * non_reload_rollouts
        + 2 * reload_rollouts,
        "optimizer_updates": len(histories),
        "replayed_examples": sum(row.replayed_examples for row in histories),
        "verifier_outcome_events": verifier_outcome_events,
        "feedback_events": unique_verifier_bits,
        "rollout_accounting": {
            "training_rollouts": training_rollouts,
            "calibration_rollouts": calibration_rollouts,
            "probe_rollouts": probe_rollouts,
            "discovery_rollouts": discovery_rollouts,
            "validation_rollouts": validation_rollouts,
            "retention_rollouts": retention_rollouts,
            "control_rollouts": control_rollouts,
            "reload_rollouts": reload_rollouts,
            "non_reload_rollouts": non_reload_rollouts,
            "total_rollouts": non_reload_rollouts + reload_rollouts,
            "unique_verifier_bits_by_phase": {
                "training": training_bits,
                "calibration": calibration_bits,
                "probe": probe_bits,
                "discovery": discovery_bits,
                "validation": validation_bits,
                "retention": retention_bits,
                "controls": control_bits,
                "reload": reload_bits,
            },
        },
        "promotion_gates": {
            "mastered_slots_protected": bool(
                candidate_status_before[0]["protected"]
                and candidate_status_before[1]["protected"]
            ),
            "unmastered_candidate_selected": selected_eviction_slot == candidate_slot,
            "stale_route_reset": route_after_eviction["preferred_slot"] is None,
            "replacement_mastered": replacement_score >= 0.8,
            "old_capabilities_retained": base_score >= 0.8 and first_score >= 0.8,
            "automatic_replacement_route": replacement_discovery_success,
            "causal_controls": controls_passed,
            "route_state_reload": reload_passed,
            "prior_mastered_bank_unchanged": prior_bank_unchanged,
            "full_protected_bank_refuses_eviction": full_bank_eviction is None,
            "controller_frozen": all(
                not parameter.requires_grad
                for parameter in agent.controller.parameters()
            ),
            "zero_replay": sum(row.replayed_examples for row in histories) == 0,
        },
        "claim_boundary": (
            "A frozen controller can reuse a bounded external capability slot "
            "only after opaque scalar evidence identifies an unprotected row; "
            "stale route evidence is cleared and mastered slots are masked. "
            "This promotes retention-safe bounded eviction, not learned general "
            "episodic utility, unbounded memory, or arbitrary new computation."
        ),
    }
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

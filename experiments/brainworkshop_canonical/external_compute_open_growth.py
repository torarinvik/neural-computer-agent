"""Promote outcome-gated, append-only growth of external compute files.

Unlike the fixed route-bank audit, this harness starts with one executable
file and allocates each later file only when a fresh candidate reaches a
stable direct mastery prefix.  The candidate is isolated from the protected
prefix, the shared controller is frozen after the first file, and the route
ledger grows only after admission.  A final same-cue probe exercises the
larger bank's ambiguity and reversal behavior.

The learner-visible boundary remains rendered events, opaque actions, and
deterministic scalar verifier outcomes.  The schedule below names private
verifier families for the experiment harness; those names never enter the
controller, external files, or route memory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch
from torch import nn

from neural_computer import (
    ExternalRegisterInstruction,
    KeypressDecoder,
    PersistentOpaqueContextRouteEvidence,
)

from .cross_family_rule_growth import RULES
from .external_compute_growth import (
    ENCODER_SYMBOL_COUNT,
    EVENT_WIDTH,
    EVENT_WINDOW_SIZE,
    INSTRUCTION_WIDTH,
    INTENTION_WIDTH,
    ComputeGrowthSystem,
    _build,
    _common_modules,
    _digest,
    _evaluate,
    _set_requires_grad,
    _slot_modules,
    _train_stage,
)
from .external_compute_route_bank import (
    ROUTE_MASTERY_THRESHOLD,
    ROUTE_SELECTION_THRESHOLD,
    _all_modules,
    _calibrate_source,
    _evaluate_route,
    _routed_episode,
    _train_route,
)

OPEN_GROWTH_SCHEMA = (
    "neural-computer.brainworkshop-external-compute-open-growth.v1"
)
OPEN_SCHEDULE = (
    ("symbol_parity", 7),
    ("triplet_parity", 8),
    ("parity2", 10),
    ("switch_binary", 11),
    ("nback2", 9),
    ("symbol_parity_odd", 5),
    ("nback3", 4),
    ("nback4", 12),
)
SOURCE_CUE = OPEN_SCHEDULE[0][1]
UNKNOWN_CUE = 6


def _append_compute_file(system: ComputeGrowthSystem, *, seed: int) -> int:
    """Append one fresh file and return its stable physical slot."""

    with torch.random.fork_rng():
        torch.manual_seed(seed)
        slot = system.machine.add_basis_slot()
        system.instructions.append(ExternalRegisterInstruction(INSTRUCTION_WIDTH))
        system.readouts.append(
            nn.Sequential(
                nn.Linear(system.machine.register_width, 16),
                nn.GELU(),
                nn.Linear(16, INTENTION_WIDTH),
            )
        )
        system.decoders.append(
            KeypressDecoder(INTENTION_WIDTH, 2, hidden=16)
        )
    return slot


def _discard_newest_compute_file(system: ComputeGrowthSystem) -> None:
    """Rollback an unadmitted candidate without touching the protected prefix."""

    if len(system.instructions) <= 1:
        raise ValueError("the source compute file cannot be discarded")
    newest = len(system.instructions) - 1
    system.machine.remove_basis_slot(newest)
    del system.instructions[newest]
    del system.readouts[newest]
    del system.decoders[newest]


def _train_and_evaluate_candidate(
    system: ComputeGrowthSystem,
    *,
    slot: int,
    family: str,
    cue_symbol: int,
    updates: int,
    batch_size: int,
    retention_lifetimes: int,
    seed: int,
    learning_rate: float,
    entropy_weight: float,
    credit_mode: str,
    shuffle_outcomes: bool = False,
) -> tuple[list[dict[str, float | int]], list[dict[str, float | int]]]:
    """Train one isolated candidate and return its history and fresh probes."""

    all_modules = _all_modules(system)
    _set_requires_grad(all_modules, False)
    candidate_modules = _slot_modules(system, slot)
    train_modules = candidate_modules
    if slot == 0:
        train_modules = _common_modules(system) + candidate_modules
    _set_requires_grad(train_modules, True)
    history = _train_stage(
        system,
        family=family,
        slot=slot,
        cue_symbol=cue_symbol,
        updates=updates,
        batch_size=batch_size,
        steps=14,
        seed=seed,
        learning_rate=learning_rate,
        entropy_weight=entropy_weight,
        credit_mode=credit_mode,
        shuffle_outcomes=shuffle_outcomes,
    )
    _set_requires_grad(all_modules, False)
    fresh = _evaluate(
        system,
        family=family,
        slot=slot,
        cue_symbol=cue_symbol,
        lifetimes=retention_lifetimes,
        batch_size=batch_size,
        steps=14,
        seed=seed + 50_000,
    )
    return history, fresh


def _stable(rows: list[dict[str, float | int]]) -> bool:
    return bool(rows) and min(
        float(row["accuracy"]) for row in rows
    ) >= ROUTE_MASTERY_THRESHOLD


def _transition_probe(
    system: ComputeGrowthSystem,
    evidence: PersistentOpaqueContextRouteEvidence,
    *,
    family: str,
    cue_symbol: int,
    slot_count: int,
    batches: int,
    seed: int,
) -> list[dict[str, object]]:
    """Probe every opaque file and update one context from scalar outcomes."""

    rows: list[dict[str, object]] = []
    for batch in range(batches):
        key, selected, accuracy, bits = _routed_episode(
            system,
            evidence,
            family=family,
            cue_symbol=cue_symbol,
            seed=seed + batch,
            slot_count=slot_count,
            exploration=0.0,
            probe_all=True,
        )
        evidence.observe_batch(key, selected, accuracy)
        slot_accuracy = {
            f"slot_{slot}_accuracy": float(accuracy[selected == slot].mean())
            for slot in range(slot_count)
        }
        slot_fractions = {
            f"slot_{slot}_fraction": float(
                (selected == slot).float().mean()
            )
            for slot in range(slot_count)
        }
        rows.append(
            {
                "batch": batch + 1,
                **slot_accuracy,
                **slot_fractions,
                "unique_verifier_bits": int(bits.sum()),
                "replayed_examples": 0,
            }
        )
    return rows


def _status(
    system: ComputeGrowthSystem,
    evidence: PersistentOpaqueContextRouteEvidence,
    cue_symbol: int,
):
    collection = system.agent.runtime.encode_streams(
        {"stimulus": torch.tensor([cue_symbol], dtype=torch.long)}
    )
    record = evidence._find_record(
        collection.payload[0, 0].detach(), create=False
    )
    if record is None:
        raise RuntimeError(f"route evidence is missing cue {cue_symbol}")
    return record.evidence.status()


def run(args: argparse.Namespace) -> dict[str, object]:
    entropy_weight = float(getattr(args, "entropy_weight", 0.01))
    credit_mode = str(getattr(args, "credit_mode", "attempted_bce"))
    event_window_size = int(
        getattr(args, "event_window_size", EVENT_WINDOW_SIZE)
    )
    if not 1 <= args.target_file_count <= len(OPEN_SCHEDULE):
        raise ValueError("target file count exceeds the calibrated schedule")
    if not args.target_file_count <= args.candidate_budget <= len(OPEN_SCHEDULE):
        raise ValueError("candidate budget must cover the target file count")
    if min(
        args.target_file_count,
        args.file_updates,
        args.route_updates,
        args.route_calibration_lifetimes,
        args.transition_batches,
        args.batch_size,
        args.retention_lifetimes,
    ) < 1:
        raise ValueError("open-growth budgets must be positive")
    if args.batch_size != 32:
        raise ValueError("the calibrated open-growth harness requires batch size 32")
    if args.learning_rate <= 0.0:
        raise ValueError("learning rate must be positive")
    if event_window_size < 1:
        raise ValueError("event window size must be positive")
    if entropy_weight < 0.0:
        raise ValueError("entropy weight cannot be negative")
    if credit_mode not in {"reinforce", "attempted_bce"}:
        raise ValueError("unsupported open-growth credit mode")
    if not 0.0 <= args.reversal_threshold <= 1.0:
        raise ValueError("reversal threshold must lie in [0, 1]")
    if args.reversal_patience < 1:
        raise ValueError("reversal patience must be positive")

    started = perf_counter()
    schedule = OPEN_SCHEDULE[: args.candidate_budget]
    system = _build(
        args.seed,
        slot_count=1,
        event_window_size=event_window_size,
    )
    controller_before = _digest(system.agent.controller)
    encoder_before = _digest(system.agent.runtime.encoders["stimulus"])
    allocation: list[dict[str, object]] = []
    histories: list[list[dict[str, float | int]]] = []
    direct: list[list[dict[str, float | int]]] = []
    file_digests: list[str] = []
    accepted_schedule: list[tuple[str, int]] = []
    attempted_training_bits = 0
    attempted_optimizer_updates = 0

    for attempt_index, (family, cue_symbol) in enumerate(schedule):
        if accepted_schedule:
            slot = _append_compute_file(
                system,
                seed=args.seed + 90_000 + attempt_index,
            )
        else:
            slot = 0
        expected_slot = len(accepted_schedule)
        if slot != expected_slot:
            raise RuntimeError("append-only compute file address skipped")
        history, fresh = _train_and_evaluate_candidate(
            system,
            slot=slot,
            family=family,
            cue_symbol=cue_symbol,
            updates=args.file_updates,
            batch_size=args.batch_size,
            retention_lifetimes=args.retention_lifetimes,
            seed=args.seed + 10_000 * (attempt_index + 1),
            learning_rate=args.learning_rate,
            entropy_weight=entropy_weight,
            credit_mode=credit_mode,
        )
        attempted_training_bits += args.batch_size * args.file_updates * (
            14 - RULES[family].warmup
        )
        attempted_optimizer_updates += args.file_updates
        accepted = _stable(fresh)
        receipt: dict[str, object] = {
            "attempt_index": attempt_index,
            "family": family,
            "cue": cue_symbol,
            "candidate_slot": slot,
            "accepted": accepted,
            "stable_direct_mastery": accepted,
            "fresh_probe": fresh,
        }
        if not accepted:
            if accepted_schedule:
                _discard_newest_compute_file(system)
                receipt["rollback"] = True
                allocation.append(receipt)
                continue
            receipt["rollback"] = False
            allocation.append(receipt)
            source_family, _source_cue = schedule[0]
            rejected_report = {
                "schema": OPEN_GROWTH_SCHEMA,
                "claim_boundary": (
                    "Outcome-gated append-only allocation of independently "
                    "trained external compute files; the source candidate "
                    "did not reach stable mastery and was rejected without "
                    "promoting a file."
                ),
                "architecture": {
                    "allocation": (
                        "fresh_direct_mastery_then_append_only_"
                        "admission_v1"
                    ),
                    "candidate_policy": "append_probe_admit_or_rollback",
                    "target_file_count": args.target_file_count,
                    "candidate_budget": args.candidate_budget,
                    "accepted_file_count": 0,
                },
                "seed": args.seed,
                "allocation": allocation,
                "gates": {
                    "source_candidate_mastered": False,
                    "rejected_candidate_not_promoted": True,
                    "zero_replayed_examples": True,
                    "frozen_controller": controller_before
                    == _digest(system.agent.controller),
                    "frozen_event_encoder": encoder_before
                    == _digest(system.agent.runtime.encoders["stimulus"]),
                },
                "accounting": {
                    "unique_verifier_bits": args.batch_size
                    * args.file_updates
                    * (14 - RULES[source_family].warmup),
                    "optimizer_updates": attempted_optimizer_updates,
                    "replayed_examples": 0,
                    "stable_bits_to_threshold": None,
                },
                "status": "rejected",
            }
            if args.report_out is not None:
                args.report_out.parent.mkdir(parents=True, exist_ok=True)
                args.report_out.write_text(
                    json.dumps(rejected_report, indent=2) + "\n"
                )
            return rejected_report
        accepted_schedule.append((family, cue_symbol))
        histories.append(history)
        direct.append(fresh)
        file_digests.append(_digest(*_slot_modules(system, slot)))
        allocation.append(receipt)
        if len(accepted_schedule) >= args.target_file_count:
            break
    accepted_count = len(accepted_schedule)
    if accepted_count < 1:
        raise RuntimeError("source compute file was not admitted")

    shuffled_control_system = _build(
        args.seed + 800_000,
        slot_count=1,
        event_window_size=event_window_size,
    )
    shuffled_control_history, shuffled_control = _train_and_evaluate_candidate(
        shuffled_control_system,
        slot=0,
        family="nback2",
        cue_symbol=9,
        updates=args.file_updates,
        batch_size=args.batch_size,
        retention_lifetimes=args.retention_lifetimes,
        seed=args.seed + 810_000,
        learning_rate=args.learning_rate,
        entropy_weight=entropy_weight,
        credit_mode=credit_mode,
        shuffle_outcomes=True,
    )

    evidence = PersistentOpaqueContextRouteEvidence(
        EVENT_WIDTH,
        matching_tolerance=1e-5,
        mastery_threshold=ROUTE_MASTERY_THRESHOLD,
        min_mastery_observations=8,
        reversal_threshold=args.reversal_threshold,
        reversal_patience=args.reversal_patience,
    )
    evidence.append_slot()
    _calibrate_source(
        system,
        evidence,
        family=accepted_schedule[0][0],
        cue_symbol=accepted_schedule[0][1],
        lifetimes=args.route_calibration_lifetimes,
        seed=args.seed + 100_000,
    )
    route_histories: list[list[dict[str, float | int]]] = [[]]
    for slot, (family, cue_symbol) in enumerate(accepted_schedule[1:], start=1):
        evidence.append_slot()
        route_histories.append(
            _train_route(
                system,
                evidence,
                family=family,
                cue_symbol=cue_symbol,
                target_slot=slot,
                updates=args.route_updates,
                seed=args.seed + 200_000 + slot * 10_000,
            )
        )

    routed = [
        _evaluate_route(
            system,
            evidence,
            family=family,
            cue_symbol=cue_symbol,
            expected_slot=slot,
            seed=args.seed + 300_000 + slot * 1_000,
            lifetimes=args.retention_lifetimes,
        )
        for slot, (family, cue_symbol) in enumerate(accepted_schedule)
    ]
    old_status_before = _status(system, evidence, SOURCE_CUE)
    transition = _transition_probe(
        system,
        evidence,
        family=accepted_schedule[-1][0],
        cue_symbol=SOURCE_CUE,
        slot_count=accepted_count,
        batches=args.transition_batches,
        seed=args.seed + 400_000,
    )
    changed_same_cue = _evaluate_route(
        system,
        evidence,
        family=accepted_schedule[-1][0],
        cue_symbol=SOURCE_CUE,
        expected_slot=accepted_count - 1,
        seed=args.seed + 500_000,
        lifetimes=args.retention_lifetimes,
    )
    old_forced = _evaluate(
        system,
        family=accepted_schedule[0][0],
        slot=0,
        cue_symbol=SOURCE_CUE,
        lifetimes=args.retention_lifetimes,
        batch_size=args.batch_size,
        steps=14,
        seed=args.seed + 600_000,
    )
    unknown = _evaluate_route(
        system,
        evidence,
        family=accepted_schedule[-1][0],
        cue_symbol=UNKNOWN_CUE,
        expected_slot=0,
        seed=args.seed + 700_000,
        lifetimes=args.retention_lifetimes,
    )
    restored = PersistentOpaqueContextRouteEvidence.from_payload(evidence.payload())
    restored_changed = _evaluate_route(
        system,
        restored,
        family=accepted_schedule[-1][0],
        cue_symbol=SOURCE_CUE,
        expected_slot=accepted_count - 1,
        seed=args.seed + 500_000,
        lifetimes=args.retention_lifetimes,
    )
    old_status_after = _status(system, evidence, SOURCE_CUE)
    controller_after = _digest(system.agent.controller)
    encoder_after = _digest(system.agent.runtime.encoders["stimulus"])
    file_digests_after = [
        _digest(*_slot_modules(system, slot)) for slot in range(accepted_count)
    ]
    old_retention_min = min(float(row["accuracy"]) for row in old_forced)
    gates = {
        "target_file_count_admitted": accepted_count >= args.target_file_count,
        "every_admitted_file_mastered": all(_stable(rows) for rows in direct),
        "every_route_mastered": all(
            float(result["accuracy"]) >= ROUTE_MASTERY_THRESHOLD
            for result in routed
        ),
        "every_route_selects_correct_file": all(
            float(result["selected_slot_fraction"]) >= ROUTE_SELECTION_THRESHOLD
            for result in routed
        ),
        "same_context_stale_route_demoted": old_status_after.reversal_count[0] >= 1,
        "same_context_replacement_preferred": changed_same_cue[
            "selected_slot_fraction"
        ]
        >= ROUTE_SELECTION_THRESHOLD,
        "same_context_replacement_mastered": changed_same_cue["accuracy"]
        >= ROUTE_MASTERY_THRESHOLD,
        "old_file_retained_after_reversal": old_retention_min
        >= ROUTE_MASTERY_THRESHOLD,
        "unknown_context_uses_oldest_fallback": unknown[
            "selected_slot_fraction"
        ]
        >= ROUTE_SELECTION_THRESHOLD,
        "unknown_context_near_chance": unknown["accuracy"] < 0.70,
        "reward_shuffled_nback2_control_rejects_mastery": max(
            float(row["accuracy"]) for row in shuffled_control
        )
        < ROUTE_MASTERY_THRESHOLD,
        "route_reload_exact": changed_same_cue == restored_changed,
        "all_admitted_files_unchanged_during_routing": file_digests
        == file_digests_after,
        "frozen_controller": controller_before == controller_after,
        "frozen_event_encoder": encoder_before == encoder_after,
        "zero_replayed_examples": True,
    }
    training_bits = attempted_training_bits
    training_bits += args.batch_size * args.route_calibration_lifetimes * (
        14 - RULES[accepted_schedule[0][0]].warmup
    )
    training_bits += args.batch_size * args.route_updates * sum(
        14 - RULES[family].warmup for family, _cue in accepted_schedule[1:]
    )
    report = {
        "schema": OPEN_GROWTH_SCHEMA,
        "claim_boundary": (
            "Outcome-gated append-only allocation of independently trained "
            "external compute files with protected-prefix retention and "
            "same-context route reversal; this remains bounded growth and "
            "does not establish arbitrary program induction or general "
            "continual learning."
        ),
        "architecture": {
            "allocation": "fresh_direct_mastery_then_append_only_admission_v1",
            "file_storage": "external_register_basis_slot_v1",
            "route_query": "learned_event_tensor_key",
            "route_feedback": "terminal_scalar_episode_accuracy",
            "route_memory": "persistent_opaque_context_route_evidence_v1",
            "candidate_policy": "append_probe_admit_or_rollback",
            "candidate_training": (
                "scalar_attempted_outcome_credit_with_optional_entropy_v1"
            ),
            "entropy_weight": entropy_weight,
            "credit_mode": credit_mode,
            "event_window_size": event_window_size,
            "encoder_symbol_count": ENCODER_SYMBOL_COUNT,
            "target_file_count": args.target_file_count,
            "candidate_budget": args.candidate_budget,
            "accepted_file_count": accepted_count,
            "candidate_attempts": len(allocation),
            "rejected_candidates": sum(
                not bool(receipt["accepted"]) for receipt in allocation
            ),
            "schedule": [
                {"family": family, "cue": cue}
                for family, cue in accepted_schedule
            ],
            "unknown_cue": UNKNOWN_CUE,
        },
        "seed": args.seed,
        "allocation": allocation,
        "histories_tail": [history[-5:] for history in histories],
        "direct": direct,
        "routed": routed,
        "route_history_tails": [history[-5:] for history in route_histories],
        "reward_shuffled_nback2_control": {
            "history_tail": shuffled_control_history[-5:],
            "evaluation": shuffled_control,
            "replayed_examples": 0,
        },
        "old_status_before": {
            "attempts": list(old_status_before.attempts),
            "protected": list(old_status_before.protected),
            "preferred_slot": old_status_before.preferred_slot,
        },
        "transition": transition,
        "old_status_after": {
            "attempts": list(old_status_after.attempts),
            "protected": list(old_status_after.protected),
            "reversal_count": list(old_status_after.reversal_count),
            "preferred_slot": old_status_after.preferred_slot,
        },
        "changed_same_cue": changed_same_cue,
        "old_forced_retention": old_forced,
        "unknown_context": unknown,
        "restored_changed_same_cue": restored_changed,
        "file_digests_before": file_digests,
        "file_digests_after": file_digests_after,
        "gates": gates,
        "accounting": {
            "unique_verifier_bits": training_bits,
            "control_verifier_bits": args.batch_size
            * args.file_updates
            * (14 - RULES["nback2"].warmup),
            "control_logical_lifetimes": args.batch_size * args.file_updates,
            "audit_verifier_bits": sum(
                int(row["unique_verifier_bits"])
                for rows in direct
                for row in rows
            )
            + sum(int(result["unique_verifier_bits"]) for result in routed)
            + sum(int(row["unique_verifier_bits"]) for row in transition)
            + sum(int(row["unique_verifier_bits"]) for row in old_forced),
            "unique_logical_lifetimes": args.batch_size
            * (
                len(allocation) * args.file_updates
                + (accepted_count - 1) * args.route_updates
                + args.route_calibration_lifetimes
                + args.transition_batches
            ),
            "optimizer_updates": attempted_optimizer_updates,
            "control_optimizer_updates": args.file_updates,
            "control_replayed_examples": 0,
            "route_memory_updates": args.route_calibration_lifetimes
            + (accepted_count - 1) * args.route_updates
            + args.transition_batches * accepted_count,
            "replayed_examples": 0,
            "latency_seconds": perf_counter() - started,
            "stable_bits_to_threshold": training_bits if all(gates.values()) else None,
        },
        "status": "promoted_external_compute_open_growth"
        if all(gates.values())
        else "rejected",
    }
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--target-file-count", type=int, default=5
    )
    parser.add_argument(
        "--candidate-budget", type=int, default=len(OPEN_SCHEDULE)
    )
    parser.add_argument("--file-updates", type=int, default=192)
    parser.add_argument("--route-updates", type=int, default=256)
    parser.add_argument("--route-calibration-lifetimes", type=int, default=8)
    parser.add_argument("--transition-batches", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--retention-lifetimes", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--event-window-size", type=int, default=EVENT_WINDOW_SIZE)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument(
        "--credit-mode",
        choices=("attempted_bce", "reinforce"),
        default="attempted_bce",
    )
    parser.add_argument("--reversal-threshold", type=float, default=0.65)
    parser.add_argument("--reversal-patience", type=int, default=4)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()

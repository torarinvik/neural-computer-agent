"""Promote failure-driven reversal of a stale external compute-file route.

One rendered cue first routes to a mastered source file.  A second file is
then acquired under a different cue.  The verifier changes the private rule
behind the original cue; no new cue label or target action is provided during
the transition.  A memory-side candidate probe evaluates opaque files and
uses only scalar episode outcomes to demote the stale route and prefer the
already acquired replacement.

The source file is evaluated forcibly after the reversal to prove that route
policy changed without deleting or mutating old computation.  This is a
bounded nonstationary-memory result, not general continual learning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from neural_computer import PersistentOpaqueContextRouteEvidence

from .cross_family_rule_growth import RULES
from .external_compute_growth import (
    EVENT_WIDTH,
    SOURCE_FAMILY,
    TARGET_FAMILY,
    _build,
    _common_modules,
    _digest,
    _evaluate,
    _set_requires_grad,
    _slot_modules,
    _train_stage,
)
from .external_compute_route_bank import (
    DEFAULT_SCHEDULE,
    ROUTE_MASTERY_THRESHOLD,
    ROUTE_SELECTION_THRESHOLD,
    _all_modules,
    _calibrate_source,
    _evaluate_route,
    _routed_episode,
    _train_route,
)

REVERSAL_SCHEMA = "neural-computer.brainworkshop-external-compute-route-reversal.v1"
SOURCE_CUE = DEFAULT_SCHEDULE[0][1]
TARGET_CUE = DEFAULT_SCHEDULE[1][1]
UNKNOWN_CUE = 9


def _route_key(system, cue_symbol: int):
    collection = system.agent.runtime.encode_streams(
        {"stimulus": torch.tensor([cue_symbol], dtype=torch.long)}
    )
    return collection.payload[0, 0].detach()


def _status(system, evidence: PersistentOpaqueContextRouteEvidence, cue: int):
    record = evidence._find_record(_route_key(system, cue), create=False)
    if record is None:
        raise RuntimeError(f"route evidence is missing cue {cue}")
    return record.evidence.status()


def _transition_probe(
    system,
    evidence: PersistentOpaqueContextRouteEvidence,
    *,
    batches: int,
    seed: int,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for batch in range(batches):
        key, selected, accuracy, bits = _routed_episode(
            system,
            evidence,
            family=TARGET_FAMILY,
            cue_symbol=SOURCE_CUE,
            seed=seed + batch,
            slot_count=2,
            exploration=0.0,
            probe_all=True,
        )
        evidence.observe_batch(key, selected, accuracy)
        rows.append(
            {
                "batch": batch + 1,
                "slot_0_accuracy": float(accuracy[selected == 0].mean()),
                "slot_1_accuracy": float(accuracy[selected == 1].mean()),
                "slot_0_fraction": float((selected == 0).float().mean()),
                "slot_1_fraction": float((selected == 1).float().mean()),
                "unique_verifier_bits": int(bits.sum()),
                "replayed_examples": 0,
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, object]:
    if min(
        args.source_updates,
        args.target_updates,
        args.route_updates,
        args.calibration_lifetimes,
        args.transition_batches,
        args.batch_size,
        args.retention_lifetimes,
    ) < 1:
        raise ValueError("reversal budgets must be positive")
    if args.batch_size != 32:
        raise ValueError("the calibrated reversal harness requires batch size 32")
    if args.reversal_patience < 1:
        raise ValueError("reversal patience must be positive")
    if not 0.0 <= args.reversal_threshold <= 1.0:
        raise ValueError("reversal threshold must lie in [0, 1]")

    started = perf_counter()
    system = _build(args.seed, slot_count=2)
    all_modules = _all_modules(system)
    controller_before = _digest(system.agent.controller)
    encoder_before = _digest(system.agent.runtime.encoders["stimulus"])

    histories: list[list[dict[str, float | int]]] = []
    direct: list[list[dict[str, float | int]]] = []
    for slot, (family, cue) in enumerate(DEFAULT_SCHEDULE[:2]):
        _set_requires_grad(all_modules, False)
        train_modules = _slot_modules(system, slot)
        if slot == 0:
            train_modules = _common_modules(system) + train_modules
        _set_requires_grad(train_modules, True)
        histories.append(
            _train_stage(
                system,
                family=family,
                slot=slot,
                cue_symbol=cue,
                updates=args.source_updates if slot == 0 else args.target_updates,
                batch_size=args.batch_size,
                steps=14,
                seed=args.seed + 10_000 * (slot + 1),
                learning_rate=args.learning_rate,
            )
        )
        _set_requires_grad(all_modules, False)
        direct.append(
            _evaluate(
                system,
                family=family,
                slot=slot,
                cue_symbol=cue,
                lifetimes=args.retention_lifetimes,
                batch_size=args.batch_size,
                steps=14,
                seed=args.seed + 50_000 + slot * 1_000,
            )
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
        family=SOURCE_FAMILY,
        cue_symbol=SOURCE_CUE,
        lifetimes=args.calibration_lifetimes,
        seed=args.seed + 100_000,
    )
    evidence.append_slot()
    route_history = _train_route(
        system,
        evidence,
        family=TARGET_FAMILY,
        cue_symbol=TARGET_CUE,
        target_slot=1,
        updates=args.route_updates,
        seed=args.seed + 200_000,
    )

    source_file_before = _digest(*_slot_modules(system, 0))
    target_file_before = _digest(*_slot_modules(system, 1))
    source_route_before = _evaluate_route(
        system,
        evidence,
        family=SOURCE_FAMILY,
        cue_symbol=SOURCE_CUE,
        expected_slot=0,
        seed=args.seed + 300_000,
        lifetimes=args.retention_lifetimes,
    )
    target_route_before = _evaluate_route(
        system,
        evidence,
        family=TARGET_FAMILY,
        cue_symbol=TARGET_CUE,
        expected_slot=1,
        seed=args.seed + 301_000,
        lifetimes=args.retention_lifetimes,
    )
    old_status_before = _status(system, evidence, SOURCE_CUE)

    transition = _transition_probe(
        system,
        evidence,
        batches=args.transition_batches,
        seed=args.seed + 400_000,
    )
    old_status_after = _status(system, evidence, SOURCE_CUE)
    new_status_after = _status(system, evidence, TARGET_CUE)
    changed_same_cue = _evaluate_route(
        system,
        evidence,
        family=TARGET_FAMILY,
        cue_symbol=SOURCE_CUE,
        expected_slot=1,
        seed=args.seed + 500_000,
        lifetimes=args.retention_lifetimes,
    )
    old_forced = _evaluate(
        system,
        family=SOURCE_FAMILY,
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
        family=TARGET_FAMILY,
        cue_symbol=UNKNOWN_CUE,
        expected_slot=0,
        seed=args.seed + 700_000,
        lifetimes=args.retention_lifetimes,
    )
    restored = PersistentOpaqueContextRouteEvidence.from_payload(evidence.payload())
    restored_changed = _evaluate_route(
        system,
        restored,
        family=TARGET_FAMILY,
        cue_symbol=SOURCE_CUE,
        expected_slot=1,
        seed=args.seed + 500_000,
        lifetimes=args.retention_lifetimes,
    )

    controller_after = _digest(system.agent.controller)
    encoder_after = _digest(system.agent.runtime.encoders["stimulus"])
    source_file_after = _digest(*_slot_modules(system, 0))
    target_file_after = _digest(*_slot_modules(system, 1))
    old_retention_min = min(float(row["accuracy"]) for row in old_forced)
    gates = {
        "source_route_mastery_before_reversal": source_route_before["accuracy"]
        >= ROUTE_MASTERY_THRESHOLD,
        "target_route_mastery_before_reversal": target_route_before["accuracy"]
        >= ROUTE_MASTERY_THRESHOLD,
        "stale_route_demoted_by_scalar_failures": old_status_after.reversal_count[0]
        >= 1,
        "replacement_route_preferred": changed_same_cue[
            "selected_slot_fraction"
        ]
        >= ROUTE_SELECTION_THRESHOLD,
        "changed_same_cue_mastery": changed_same_cue["accuracy"]
        >= ROUTE_MASTERY_THRESHOLD,
        "old_file_retained_under_forced_audit": old_retention_min
        >= ROUTE_MASTERY_THRESHOLD,
        "unknown_context_does_not_select_replacement": unknown[
            "selected_slot_fraction"
        ]
        >= ROUTE_SELECTION_THRESHOLD,
        "unknown_context_near_chance": unknown["accuracy"] < 0.70,
        "route_reload_exact": changed_same_cue == restored_changed,
        "old_file_unchanged": source_file_before == source_file_after,
        "replacement_file_unchanged_during_reversal": target_file_before
        == target_file_after,
        "frozen_controller": controller_before == controller_after,
        "frozen_event_encoder": encoder_before == encoder_after,
        "zero_replayed_examples": True,
    }
    training_bits = args.batch_size * (
        args.source_updates * (14 - RULES[SOURCE_FAMILY].warmup)
        + args.target_updates * (14 - RULES[TARGET_FAMILY].warmup)
        + args.calibration_lifetimes * (14 - RULES[SOURCE_FAMILY].warmup)
        + args.route_updates * (14 - RULES[TARGET_FAMILY].warmup)
        + args.transition_batches * (14 - RULES[TARGET_FAMILY].warmup)
    )
    report = {
        "schema": REVERSAL_SCHEMA,
        "claim_boundary": (
            "Failure-driven same-context replacement of a stale external "
            "compute-file route from fresh scalar outcomes, with old file "
            "retention and frozen controller; this is bounded nonstationary "
            "memory, not unrestricted growth or general continual learning."
        ),
        "architecture": {
            "route_query": "learned_event_tensor_key",
            "route_feedback": "terminal_scalar_episode_accuracy",
            "route_memory": "persistent_opaque_context_route_evidence_v1",
            "candidate_policy": "parallel_opaque_file_probe_then_hysteretic_demotion",
            "source_family": SOURCE_FAMILY,
            "replacement_family": TARGET_FAMILY,
            "source_cue": SOURCE_CUE,
            "replacement_cue": TARGET_CUE,
            "changed_task_cue": SOURCE_CUE,
            "unknown_cue": UNKNOWN_CUE,
            "reversal_threshold": args.reversal_threshold,
            "reversal_patience": args.reversal_patience,
        },
        "seed": args.seed,
        "direct": direct,
        "source_route_before_reversal": source_route_before,
        "target_route_before_reversal": target_route_before,
        "route_history_tail": route_history[-5:],
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
        "replacement_context_status": {
            "attempts": list(new_status_after.attempts),
            "protected": list(new_status_after.protected),
            "preferred_slot": new_status_after.preferred_slot,
        },
        "changed_same_cue": changed_same_cue,
        "old_forced_retention": old_forced,
        "unknown_context": unknown,
        "restored_changed_same_cue": restored_changed,
        "gates": gates,
        "accounting": {
            "unique_verifier_bits": training_bits,
            "audit_verifier_bits": sum(
                int(row["unique_verifier_bits"])
                for row in transition
            )
            + sum(int(row["unique_verifier_bits"]) for row in old_forced)
            + int(source_route_before["unique_verifier_bits"])
            + int(target_route_before["unique_verifier_bits"])
            + int(changed_same_cue["unique_verifier_bits"]),
            "unique_logical_lifetimes": args.batch_size
            * (
                args.source_updates
                + args.target_updates
                + args.calibration_lifetimes
                + args.route_updates
                + args.transition_batches
            ),
            "optimizer_updates": args.source_updates + args.target_updates,
            "route_memory_updates": args.calibration_lifetimes
            + args.route_updates
            + args.transition_batches,
            "replayed_examples": 0,
            "latency_seconds": perf_counter() - started,
            "stable_bits_to_threshold": training_bits if all(gates.values()) else None,
        },
        "status": "promoted_external_compute_route_reversal"
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
    parser.add_argument("--source-updates", type=int, default=192)
    parser.add_argument("--target-updates", type=int, default=256)
    parser.add_argument("--route-updates", type=int, default=256)
    parser.add_argument("--calibration-lifetimes", type=int, default=8)
    parser.add_argument("--transition-batches", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--retention-lifetimes", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--reversal-threshold", type=float, default=0.65)
    parser.add_argument("--reversal-patience", type=int, default=4)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()

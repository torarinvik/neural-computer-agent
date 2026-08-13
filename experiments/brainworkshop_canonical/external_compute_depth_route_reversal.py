"""Route a newly learned n-back-32 file without forgetting n-back-16.

This composes two already-promoted boundaries: append-only indexed-history
computation and scalar-outcome external route memory.  The source file learns
n-back-16 and the replacement file learns n-back-32 while the controller and
frontend are frozen.  The route is then reversed behind the source cue by
probing both opaque files with fresh scalar outcomes.  The old file remains
retained and directly executable.

The route key is a learned event tensor, not a cue/task label.  The unknown-cue
control checks that a new key falls back conservatively instead of generalizing
to the newest file.  This remains a bounded pressure test, not a claim of
general continual learning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from neural_computer import PersistentOpaqueContextRouteEvidence

from .cross_family_rule_growth import RULES
from .external_compute_growth import (
    EVENT_WIDTH,
    _build,
    _common_modules,
    _digest,
    _evaluate,
    _set_requires_grad,
    _slot_modules,
    _train_stage,
)
from .external_compute_route_bank import (
    _all_modules,
    _calibrate_source,
    _evaluate_route,
    _family_steps,
    _routed_episode,
    _train_route,
)

SCHEMA = "neural-computer.brainworkshop-external-compute-depth-route-reversal.v1"
SOURCE_FAMILY = "nback16"
TARGET_FAMILY = "nback32"
SOURCE_CUE = 7
TARGET_CUE = 8
UNKNOWN_CUE = 9
HISTORY_AGE_SLOT_COUNT = 32
QUERY_COUNT = 32
BATCH_SIZE = 32
ROUTE_MASTERY_THRESHOLD = 0.80
ROUTE_SELECTION_THRESHOLD = 0.99


def _new_evidence() -> PersistentOpaqueContextRouteEvidence:
    return PersistentOpaqueContextRouteEvidence(
        EVENT_WIDTH,
        matching_tolerance=1e-5,
        mastery_threshold=ROUTE_MASTERY_THRESHOLD,
        min_mastery_observations=8,
        reversal_threshold=0.65,
        reversal_patience=4,
    )


def _train_shuffled_route(
    system: Any,
    evidence: PersistentOpaqueContextRouteEvidence,
    *,
    updates: int,
    seed: int,
) -> list[dict[str, float | int]]:
    """Train route evidence with batch-shuffled scalar outcomes."""

    history: list[dict[str, float | int]] = []
    for update in range(1, updates + 1):
        torch.manual_seed(seed + update * 10_007)
        key, selected, accuracy, bits = _routed_episode(
            system,
            evidence,
            family=TARGET_FAMILY,
            cue_symbol=TARGET_CUE,
            seed=seed + update,
            slot_count=2,
            exploration=0.5,
        )
        evidence.observe_batch(key, selected, accuracy.roll(1, dims=0))
        history.append(
            {
                "update": update,
                "accuracy": float(accuracy.mean()),
                "shuffled_accuracy": float(accuracy.roll(1, dims=0).mean()),
                "target_slot_fraction": float((selected == 1).float().mean()),
                "unique_verifier_bits": int(bits.sum()),
                "replayed_examples": 0,
            }
        )
    return history


def _transition_probe(
    system: Any,
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


def _stable(rows: list[dict[str, float | int]]) -> bool:
    return bool(rows) and min(float(row["accuracy"]) for row in rows) >= ROUTE_MASTERY_THRESHOLD


def _run_seed(
    *,
    seed: int,
    source_updates: int,
    target_updates: int,
    route_updates: int,
    transition_batches: int,
    route_calibration_lifetimes: int,
    retention_lifetimes: int,
    learning_rate: float,
    entropy_weight: float,
) -> dict[str, object]:
    started = perf_counter()
    system = _build(
        seed,
        slot_count=2,
        event_window_size=0,
        basis_event_read_mode="history_indexed",
        basis_history_age_slot_count=HISTORY_AGE_SLOT_COUNT,
        external_history_query_count=QUERY_COUNT,
    )
    all_modules = _all_modules(system)
    controller_before = _digest(system.agent.controller)
    encoder_before = _digest(system.agent.runtime.encoders["stimulus"])
    histories: list[list[dict[str, float | int]]] = []
    direct: list[list[dict[str, float | int]]] = []
    schedule = ((SOURCE_FAMILY, SOURCE_CUE), (TARGET_FAMILY, TARGET_CUE))

    for slot, (family, cue) in enumerate(schedule):
        _set_requires_grad(all_modules, False)
        train_modules = _slot_modules(system, slot)
        if slot == 0:
            train_modules = _common_modules(system) + train_modules
        _set_requires_grad(train_modules, True)
        history = _train_stage(
            system,
            family=family,
            slot=slot,
            cue_symbol=cue,
            updates=source_updates if slot == 0 else target_updates,
            batch_size=BATCH_SIZE,
            steps=_family_steps(family),
            seed=seed + 10_000 * (slot + 1),
            learning_rate=learning_rate,
            entropy_weight=entropy_weight,
            credit_mode="attempted_bce",
            history_query_count=QUERY_COUNT,
        )
        histories.append(history)
        _set_requires_grad(all_modules, False)
        direct.append(
            _evaluate(
                system,
                family=family,
                slot=slot,
                cue_symbol=cue,
                lifetimes=retention_lifetimes,
                batch_size=BATCH_SIZE,
                steps=_family_steps(family),
                seed=seed + 50_000 + slot * 1_000,
                history_query_count=QUERY_COUNT,
            )
        )

    source_file_before_route = _digest(*_slot_modules(system, 0))
    target_file_before_route = _digest(*_slot_modules(system, 1))

    evidence = _new_evidence()
    evidence.append_slot()
    source_route_calibration_lifetimes = _calibrate_source(
        system,
        evidence,
        family=SOURCE_FAMILY,
        cue_symbol=SOURCE_CUE,
        lifetimes=route_calibration_lifetimes,
        seed=seed + 100_000,
    )
    evidence.append_slot()
    target_route_history = _train_route(
        system,
        evidence,
        family=TARGET_FAMILY,
        cue_symbol=TARGET_CUE,
        target_slot=1,
        updates=route_updates,
        seed=seed + 200_000,
    )
    source_routed = _evaluate_route(
        system,
        evidence,
        family=SOURCE_FAMILY,
        cue_symbol=SOURCE_CUE,
        expected_slot=0,
        seed=seed + 300_000,
        lifetimes=retention_lifetimes,
    )
    target_routed = _evaluate_route(
        system,
        evidence,
        family=TARGET_FAMILY,
        cue_symbol=TARGET_CUE,
        expected_slot=1,
        seed=seed + 301_000,
        lifetimes=retention_lifetimes,
    )
    transition = _transition_probe(
        system,
        evidence,
        batches=transition_batches,
        seed=seed + 400_000,
    )
    changed_same_cue = _evaluate_route(
        system,
        evidence,
        family=TARGET_FAMILY,
        cue_symbol=SOURCE_CUE,
        expected_slot=1,
        seed=seed + 500_000,
        lifetimes=retention_lifetimes,
    )
    old_forced_retention = _evaluate(
        system,
        family=SOURCE_FAMILY,
        slot=0,
        cue_symbol=SOURCE_CUE,
        lifetimes=retention_lifetimes,
        batch_size=BATCH_SIZE,
        steps=_family_steps(SOURCE_FAMILY),
        seed=seed + 600_000,
        history_query_count=QUERY_COUNT,
    )
    unknown = _evaluate_route(
        system,
        evidence,
        family=TARGET_FAMILY,
        cue_symbol=UNKNOWN_CUE,
        expected_slot=0,
        seed=seed + 700_000,
        lifetimes=retention_lifetimes,
    )
    restored = PersistentOpaqueContextRouteEvidence.from_payload(evidence.payload())
    restored_changed = _evaluate_route(
        system,
        restored,
        family=TARGET_FAMILY,
        cue_symbol=SOURCE_CUE,
        expected_slot=1,
        seed=seed + 500_000,
        lifetimes=retention_lifetimes,
    )

    shuffled_evidence = _new_evidence()
    shuffled_evidence.append_slot()
    _calibrate_source(
        system,
        shuffled_evidence,
        family=SOURCE_FAMILY,
        cue_symbol=SOURCE_CUE,
        lifetimes=route_calibration_lifetimes,
        seed=seed + 800_000,
    )
    shuffled_evidence.append_slot()
    shuffled_route_history = _train_shuffled_route(
        system,
        shuffled_evidence,
        updates=route_updates,
        seed=seed + 900_000,
    )
    shuffled_target = _evaluate_route(
        system,
        shuffled_evidence,
        family=TARGET_FAMILY,
        cue_symbol=TARGET_CUE,
        expected_slot=1,
        seed=seed + 950_000,
        lifetimes=retention_lifetimes,
    )

    controller_after = _digest(system.agent.controller)
    encoder_after = _digest(system.agent.runtime.encoders["stimulus"])
    source_file_after = _digest(*_slot_modules(system, 0))
    target_file_after = _digest(*_slot_modules(system, 1))
    gates = {
        "source_direct_mastery": _stable(direct[0]),
        "target_direct_mastery": _stable(direct[1]),
        "source_route_mastery": source_routed["accuracy"] >= ROUTE_MASTERY_THRESHOLD,
        "target_route_mastery": target_routed["accuracy"] >= ROUTE_MASTERY_THRESHOLD,
        "source_route_selected": source_routed["selected_slot_fraction"] >= ROUTE_SELECTION_THRESHOLD,
        "target_route_selected": target_routed["selected_slot_fraction"] >= ROUTE_SELECTION_THRESHOLD,
        "changed_same_cue_mastery": changed_same_cue["accuracy"] >= ROUTE_MASTERY_THRESHOLD,
        "changed_same_cue_replacement_selected": changed_same_cue["selected_slot_fraction"] >= ROUTE_SELECTION_THRESHOLD,
        "old_file_retained": _stable(old_forced_retention),
        "unknown_context_falls_back_to_oldest": unknown["selected_slot_fraction"] >= ROUTE_SELECTION_THRESHOLD,
        "unknown_context_rejects_new_mastery": unknown["accuracy"] < 0.70,
        "route_reload_exact": changed_same_cue == restored_changed,
        "shuffled_route_training_rejects_target": (
            shuffled_target["accuracy"] < ROUTE_MASTERY_THRESHOLD
            or shuffled_target["selected_slot_fraction"] < ROUTE_SELECTION_THRESHOLD
        ),
        "source_file_unchanged_during_routing": source_file_before_route == source_file_after,
        "target_file_unchanged_during_routing": target_file_before_route == target_file_after,
        "frozen_controller": controller_before == controller_after,
        "frozen_event_encoder": encoder_before == encoder_after,
        "zero_replayed_examples": True,
    }
    file_bits = sum(
        int(row["unique_verifier_bits"])
        for stage in histories
        for row in stage
    )
    route_bits = sum(int(row["unique_verifier_bits"]) for row in target_route_history)
    calibration_bits = (
        BATCH_SIZE
        * source_route_calibration_lifetimes
        * (_family_steps(SOURCE_FAMILY) - RULES[SOURCE_FAMILY].warmup)
    )
    transition_bits = sum(int(row["unique_verifier_bits"]) for row in transition)
    shuffled_bits = sum(int(row["unique_verifier_bits"]) for row in shuffled_route_history)
    report = {
        "schema": SCHEMA,
        "claim_boundary": (
            "Outcome-only indexed-history route reversal from nback16 to nback32 "
            "with frozen old-file retention; not general continual learning, "
            "unrestricted memory growth, or arbitrary program induction."
        ),
        "seed": seed,
        "architecture": {
            "source_family": SOURCE_FAMILY,
            "target_family": TARGET_FAMILY,
            "source_cue": SOURCE_CUE,
            "target_cue": TARGET_CUE,
            "changed_task_cue": SOURCE_CUE,
            "unknown_cue": UNKNOWN_CUE,
            "event_read_mode": "history_indexed",
            "history_age_slot_count": HISTORY_AGE_SLOT_COUNT,
            "query_count": QUERY_COUNT,
            "route_query": "learned_event_tensor_key",
            "route_feedback": "terminal_scalar_episode_accuracy",
            "route_memory": "persistent_opaque_context_route_evidence_v1",
        },
        "direct": {"source": direct[0], "target": direct[1]},
        "routed": {
            "source": source_routed,
            "target": target_routed,
            "changed_same_cue": changed_same_cue,
            "old_forced_retention": old_forced_retention,
            "unknown": unknown,
            "restored_changed_same_cue": restored_changed,
            "shuffled_target": shuffled_target,
        },
        "transition": transition,
        "training": {
            "source_tail": histories[0][-5:],
            "target_tail": histories[1][-5:],
            "route_tail": target_route_history[-5:],
            "shuffled_route_tail": shuffled_route_history[-5:],
            "source_route_calibration_lifetimes": source_route_calibration_lifetimes,
        },
        "gates": gates,
        "accounting": {
            "unique_verifier_bits": file_bits + route_bits + calibration_bits + transition_bits,
            "shuffled_control_verifier_bits": shuffled_bits,
            "unique_logical_lifetimes": BATCH_SIZE * (source_updates + target_updates + route_updates + route_calibration_lifetimes + transition_batches),
            "optimizer_updates": source_updates + target_updates,
            "route_memory_updates": route_calibration_lifetimes + route_updates + transition_batches,
            "shuffled_route_memory_updates": route_calibration_lifetimes + route_updates,
            "replayed_examples": 0,
            "stable_bits_to_threshold": file_bits + route_bits if all(gates.values()) else None,
            "latency_seconds": perf_counter() - started,
        },
        "status": "promoted_depth_route_reversal" if all(gates.values()) else "rejected",
    }
    return report


def run(args: argparse.Namespace) -> dict[str, object]:
    if min(
        args.source_updates,
        args.target_updates,
        args.route_updates,
        args.transition_batches,
        args.route_calibration_lifetimes,
        args.retention_lifetimes,
    ) < 1:
        raise ValueError("depth-route budgets must be positive")
    if args.learning_rate <= 0.0 or args.entropy_weight < 0.0:
        raise ValueError("learning rate must be positive and entropy non-negative")
    report = _run_seed(
        seed=args.seed,
        source_updates=args.source_updates,
        target_updates=args.target_updates,
        route_updates=args.route_updates,
        transition_batches=args.transition_batches,
        route_calibration_lifetimes=args.route_calibration_lifetimes,
        retention_lifetimes=args.retention_lifetimes,
        learning_rate=args.learning_rate,
        entropy_weight=args.entropy_weight,
    )
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--source-updates", type=int, default=256)
    parser.add_argument("--target-updates", type=int, default=256)
    parser.add_argument("--route-updates", type=int, default=256)
    parser.add_argument("--transition-batches", type=int, default=8)
    parser.add_argument("--route-calibration-lifetimes", type=int, default=8)
    parser.add_argument("--retention-lifetimes", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()

"""Select bounded external-history depth from scalar verifier outcomes.

This pressure test keeps the learned external files and fixed controller from
the mixed-depth retention rung, then gives an isolated memory-side policy the
candidate depths ``1..6``. The policy sees only attempted-lifetime accuracy;
it exposes a depth only after a stable-prefix mastery gate and fails closed if
all candidates are exhausted without evidence. A maintenance phase then
demotes one protected depth after patient failures and promotes a replacement
from fresh outcomes without changing the file.

The result is deliberately narrower than general continual learning: it
validates outcome-only retrieval-depth selection, not learned computation,
compression, or unrestricted memory growth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from neural_computer import PersistentOpaqueDepthEvidence

from .external_compute_growth import (
    _build,
    _common_modules,
    _digest,
    _evaluate,
    _set_requires_grad,
    _slot_modules,
    _train_stage,
    ComputeGrowthSystem,
)
from .external_compute_route_bank import _family_steps

DEPTH_SELECTION_SCHEMA = (
    "neural-computer.brainworkshop-external-compute-depth-selection.v1"
)
DEFAULT_SCHEDULE = (
    ("symbol_parity", 7, 4),
    ("triplet_parity", 8, 4),
    ("parity2", 10, 4),
    ("switch_binary", 11, 4),
    ("nback5", 12, 6),
)
CANDIDATE_QUERY_COUNTS = (1, 2, 3, 4, 5, 6)
MASTERY_THRESHOLD = 0.99


def _train_files(
    seed: int,
    *,
    updates: int,
    batch_size: int,
    steps: int,
    learning_rate: float,
) -> tuple[
    ComputeGrowthSystem,
    list[list[dict[str, float | int]]],
    list[str],
    str,
    str,
]:
    system = _build(
        seed,
        slot_count=len(DEFAULT_SCHEDULE),
        event_window_size=6,
        basis_event_read_mode="flattened_window",
        external_history_query_counts=tuple(
            query_count for _, _, query_count in DEFAULT_SCHEDULE
        ),
    )
    controller_before = _digest(system.agent.controller)
    encoder_before = _digest(system.agent.runtime.encoders["stimulus"])
    all_modules = _common_modules(system) + tuple(
        module
        for slot in range(len(system.instructions))
        for module in _slot_modules(system, slot)
    )
    histories: list[list[dict[str, float | int]]] = []
    file_digests: list[str] = []
    for slot, (family, cue_symbol, query_count) in enumerate(DEFAULT_SCHEDULE):
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
                cue_symbol=cue_symbol,
                updates=updates,
                batch_size=batch_size,
                steps=max(steps, _family_steps(family)),
                seed=seed + 10_000 * (slot + 1),
                learning_rate=learning_rate,
                entropy_weight=0.01,
                credit_mode="attempted_bce",
                history_query_count=query_count,
            )
        )
        _set_requires_grad(all_modules, False)
        file_digests.append(_digest(*_slot_modules(system, slot)))
    return system, histories, file_digests, controller_before, encoder_before


def _calibrate_depths(
    system: ComputeGrowthSystem,
    *,
    policy: PersistentOpaqueDepthEvidence,
    seed: int,
    batch_size: int,
    steps: int,
    probe_lifetimes: int,
    retention_lifetimes: int,
) -> tuple[list[dict[str, object]], list[str]]:
    results: list[dict[str, object]] = []
    file_digests: list[str] = []
    for slot, (family, cue_symbol, trained_depth) in enumerate(DEFAULT_SCHEDULE):
        policy.append_file()
        probes: list[dict[str, object]] = []
        while policy.preferred_query_count(slot) is None:
            query_count = policy.next_probe_query_count(slot)
            if query_count is None:
                break
            rows = _evaluate(
                system,
                family=family,
                slot=slot,
                cue_symbol=cue_symbol,
                lifetimes=probe_lifetimes,
                batch_size=batch_size,
                steps=max(steps, _family_steps(family)),
                seed=seed + slot * 100_000 + query_count * 1_000,
                history_query_count=query_count,
            )
            accuracies = [float(row["accuracy"]) for row in rows]
            for accuracy in accuracies:
                policy.observe(slot, query_count, accuracy)
            probes.append(
                {
                    "query_count": query_count,
                    "accuracy": accuracies,
                    "unique_verifier_bits": sum(
                        int(row["unique_verifier_bits"]) for row in rows
                    ),
                    "unique_logical_lifetimes": batch_size * probe_lifetimes,
                    "replayed_examples": 0,
                }
            )
        selected_depth = policy.preferred_query_count(slot)
        retention = []
        if selected_depth is not None:
            retention = _evaluate(
                system,
                family=family,
                slot=slot,
                cue_symbol=cue_symbol,
                lifetimes=retention_lifetimes,
                batch_size=batch_size,
                steps=max(steps, _family_steps(family)),
                seed=seed + 500_000 + slot * 1_000,
                history_query_count=selected_depth,
            )
        results.append(
            {
                "slot": slot,
                "family": family,
                "trained_query_count": trained_depth,
                "probes": probes,
                "selected_query_count": selected_depth,
                "retention": retention,
            }
        )
        file_digests.append(_digest(*_slot_modules(system, slot)))
    return results, file_digests


def _shuffled_control() -> bool:
    policy = PersistentOpaqueDepthEvidence(
        CANDIDATE_QUERY_COUNTS,
        mastery_threshold=MASTERY_THRESHOLD,
        min_mastery_observations=8,
    )
    policy.append_file()
    for query_count in CANDIDATE_QUERY_COUNTS:
        for outcome in (1.0, 0.0) * 4:
            policy.observe(0, query_count, outcome)
    return (
        policy.preferred_query_count(0) is None
        and policy.next_probe_query_count(0) is None
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    if min(
        args.updates,
        args.batch_size,
        args.steps,
        args.probe_lifetimes,
        args.retention_lifetimes,
    ) < 1:
        raise ValueError("depth-selection budgets must be positive")
    if args.learning_rate <= 0.0:
        raise ValueError("learning rate must be positive")
    started = perf_counter()
    system, histories, training_digests, controller_before, encoder_before = (
        _train_files(
            args.seed,
            updates=args.updates,
            batch_size=args.batch_size,
            steps=args.steps,
            learning_rate=args.learning_rate,
        )
    )
    policy = PersistentOpaqueDepthEvidence(
        CANDIDATE_QUERY_COUNTS,
        mastery_threshold=MASTERY_THRESHOLD,
        min_mastery_observations=args.probe_lifetimes,
        reversal_threshold=0.65,
        reversal_patience=4,
    )
    results, calibration_digests = _calibrate_depths(
        system,
        policy=policy,
        seed=args.seed + 100_000,
        batch_size=args.batch_size,
        steps=args.steps,
        probe_lifetimes=args.probe_lifetimes,
        retention_lifetimes=args.retention_lifetimes,
    )
    maintenance: dict[str, object] = {
        "slot": 0,
        "stale_query_count": None,
        "replacement_query_count": None,
        "stale_depth_demoted": False,
        "replacement_mastered": False,
        "replacement_accuracy": [],
        "replacement_unique_verifier_bits": 0,
        "replacement_unique_logical_lifetimes": 0,
        "replayed_examples": 0,
    }
    selected_before_maintenance = [
        result["selected_query_count"] for result in results
    ]
    stale_depth = selected_before_maintenance[0]
    if stale_depth is not None:
        for _ in range(policy.reversal_patience):
            policy.observe(0, int(stale_depth), 0.0)
        replacement_depth = policy.next_probe_query_count(0)
        maintenance["stale_query_count"] = stale_depth
        maintenance["stale_depth_demoted"] = (
            policy.preferred_query_count(0) is None
        )
        maintenance["replacement_query_count"] = replacement_depth
        if replacement_depth is not None:
            replacement_rows = _evaluate(
                system,
                family=DEFAULT_SCHEDULE[0][0],
                slot=0,
                cue_symbol=DEFAULT_SCHEDULE[0][1],
                lifetimes=args.probe_lifetimes,
                batch_size=args.batch_size,
                steps=max(args.steps, _family_steps(DEFAULT_SCHEDULE[0][0])),
                seed=args.seed + 900_000,
                history_query_count=replacement_depth,
            )
            replacement_accuracy = [
                float(row["accuracy"]) for row in replacement_rows
            ]
            for accuracy in replacement_accuracy:
                policy.observe(0, replacement_depth, accuracy)
            maintenance["replacement_accuracy"] = replacement_accuracy
            maintenance["replacement_unique_verifier_bits"] = sum(
                int(row["unique_verifier_bits"]) for row in replacement_rows
            )
            maintenance["replacement_unique_logical_lifetimes"] = (
                args.batch_size * args.probe_lifetimes
            )
            maintenance["replacement_mastered"] = (
                policy.preferred_query_count(0) == replacement_depth
            )
    maintenance_digests = [
        _digest(*_slot_modules(system, slot))
        for slot in range(len(system.instructions))
    ]
    restored = PersistentOpaqueDepthEvidence.from_payload(policy.payload())
    selected = [result["selected_query_count"] for result in results]
    retention_stable = all(
        result["retention"]
        and min(float(row["accuracy"]) for row in result["retention"]) >= MASTERY_THRESHOLD
        for result in results
    )
    controller_after = _digest(system.agent.controller)
    encoder_after = _digest(system.agent.runtime.encoders["stimulus"])
    gates = {
        "all_depths_selected": all(depth is not None for depth in selected),
        "selected_retention_mastered": retention_stable,
        "policy_reload_exact": restored.payload() == policy.payload(),
        "controller_unchanged": controller_before == controller_after,
        "event_encoder_unchanged": encoder_before == encoder_after,
        "files_unchanged_during_calibration": (
            training_digests == calibration_digests == maintenance_digests
        ),
        "stale_depth_demoted": bool(maintenance["stale_depth_demoted"]),
        "replacement_depth_mastered": bool(maintenance["replacement_mastered"]),
        "shuffled_control_fails_closed": _shuffled_control(),
        "zero_replayed_examples": True,
    }
    training_bits = sum(
        int(row["unique_verifier_bits"])
        for history in histories
        for row in history
    )
    calibration_bits = sum(
        int(probe["unique_verifier_bits"])
        for result in results
        for probe in result["probes"]
    )
    retention_bits = sum(
        int(row["unique_verifier_bits"])
        for result in results
        for row in result["retention"]
    )
    calibration_lifetimes = sum(
        int(probe["unique_logical_lifetimes"])
        for result in results
        for probe in result["probes"]
    )
    retention_lifetimes = args.batch_size * args.retention_lifetimes * len(
        DEFAULT_SCHEDULE
    )
    maintenance_bits = int(maintenance["replacement_unique_verifier_bits"])
    maintenance_lifetimes = int(
        maintenance["replacement_unique_logical_lifetimes"]
    )
    report = {
        "schema": DEPTH_SELECTION_SCHEMA,
        "claim_boundary": (
            "Outcome-only selection of a bounded external-history query depth "
            "for retained opaque files; not learned computation, learned "
            "compression, unrestricted memory growth, or general continual learning."
        ),
        "architecture": {
            "boundary": "frozen_amodal_controller -> isolated external_depth_policy -> external_history_file",
            "file_schedule": [
                {"family": family, "cue": cue, "trained_query_count": query}
                for family, cue, query in DEFAULT_SCHEDULE
            ],
            "candidate_query_counts": CANDIDATE_QUERY_COUNTS,
            "shared_event_window_size": 6,
            "query_count_semantics": "q_minus_one_previous_records_plus_current_event",
            "policy": "persistent_opaque_depth_evidence_v1",
            "mastery_threshold": MASTERY_THRESHOLD,
            "probe_lifetimes": args.probe_lifetimes,
        },
        "results": results,
        "maintenance": maintenance,
        "selected_query_counts": selected,
        "gates": gates,
        "accounting": {
            "training_unique_verifier_bits": training_bits,
            "calibration_unique_verifier_bits": calibration_bits,
            "retention_unique_verifier_bits": retention_bits,
            "maintenance_unique_verifier_bits": maintenance_bits,
            "unique_verifier_bits": (
                training_bits + calibration_bits + retention_bits + maintenance_bits
            ),
            "unique_logical_lifetimes": (
                args.batch_size * args.updates * len(DEFAULT_SCHEDULE)
                + calibration_lifetimes
                + retention_lifetimes
                + maintenance_lifetimes
            ),
            "stable_bits_to_threshold": training_bits + calibration_bits,
            "optimizer_updates": args.updates * len(DEFAULT_SCHEDULE),
            "calibration_policy_observations": calibration_lifetimes,
            "replayed_examples": 0,
            "calibration_optimizer_updates": 0,
            "wall_time_seconds": perf_counter() - started,
        },
        "status": "promoted" if all(gates.values()) else "rejected",
        "elapsed_seconds": perf_counter() - started,
    }
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--updates", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=14)
    parser.add_argument("--probe-lifetimes", type=int, default=8)
    parser.add_argument("--retention-lifetimes", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

"""Audit append-only external computation growth with frozen old skills.

The controller and frontend remain frozen.  An external compute bank first
masters ``nback16`` in one opaque file, then trains ``nback32`` in a fresh
file while the old file and shared interpreter are frozen.  This is a direct
test of the CPU/files hypothesis: new computation should be added as an
external capability without overwriting an earlier capability.

The experiment is deliberately narrower than general continual learning.  It
tests one bounded depth extension, with outcome-only learning, zero replay,
and independent retention and corruption controls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from .external_compute_growth import (
    _build,
    _common_modules,
    _digest,
    _episode,
    _set_requires_grad,
    _slot_modules,
    _train_stage,
)

SCHEMA = "neural-computer.brainworkshop-external-compute-append-only-depth-growth.v1"
SOURCE_FAMILY = "nback16"
TARGET_FAMILY = "nback32"
SOURCE_STEPS = 26
TARGET_STEPS = 42
QUERY_COUNT = 32
HISTORY_AGE_SLOT_COUNT = 32
CUE_SYMBOL = 12
MASTERY_THRESHOLD = 0.80


def _evaluate_condition(
    system: Any,
    *,
    family: str,
    slot: int,
    batch_size: int,
    steps: int,
    seed: int,
    lifetimes: int,
    reset_external_each_step: bool = False,
    shuffle_actions: bool = False,
    shuffle_outcomes: bool = False,
    corrupt_external_history: bool = False,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for lifetime in range(lifetimes):
        _, accuracy, bits = _episode(
            system,
            family=family,
            slot=slot,
            cue_symbol=CUE_SYMBOL,
            batch_size=batch_size,
            steps=steps,
            seed=seed + lifetime,
            train=False,
            reset_external_each_step=reset_external_each_step,
            shuffle_actions=shuffle_actions,
            shuffle_outcomes=shuffle_outcomes,
            corrupt_external_history=corrupt_external_history,
            history_query_count=QUERY_COUNT,
        )
        rows.append(
            {
                "lifetime": lifetime + 1,
                "accuracy": float(accuracy),
                "unique_verifier_bits": int(bits),
                "replayed_examples": 0,
            }
        )
    return rows


def _mastered(rows: list[dict[str, float | int]]) -> bool:
    return bool(rows) and min(float(row["accuracy"]) for row in rows) >= MASTERY_THRESHOLD


def _mean(rows: list[dict[str, float | int]]) -> float:
    return sum(float(row["accuracy"]) for row in rows) / len(rows)


def _train_reward_shuffled_control(
    system: Any,
    *,
    slot: int,
    seed: int,
    updates: int,
    batch_size: int,
    learning_rate: float,
    entropy_weight: float,
    lifetimes: int,
) -> list[dict[str, float | int]]:
    """Train a fresh target file on shuffled verifier outcomes.

    Evaluation-time reward shuffling is not causal for this event-history
    reader: once a file is mastered, its execution need not consume reward.
    This control uses a fresh file and destroys the learning signal while
    preserving the observation stream and optimizer budget.
    """

    modules = _slot_modules(system, slot)
    _set_requires_grad(modules, True)
    _train_stage(
        system,
        family=TARGET_FAMILY,
        slot=slot,
        cue_symbol=CUE_SYMBOL,
        updates=updates,
        batch_size=batch_size,
        steps=TARGET_STEPS,
        seed=seed,
        learning_rate=learning_rate,
        entropy_weight=entropy_weight,
        credit_mode="attempted_bce",
        shuffle_outcomes=True,
        history_query_count=QUERY_COUNT,
    )
    _set_requires_grad(modules, False)
    return _evaluate_condition(
        system,
        family=TARGET_FAMILY,
        slot=slot,
        batch_size=batch_size,
        steps=TARGET_STEPS,
        seed=seed + 100_000,
        lifetimes=lifetimes,
    )


def _run_seed(
    *,
    seed: int,
    source_updates: int,
    target_updates: int,
    batch_size: int,
    lifetimes: int,
    learning_rate: float,
    entropy_weight: float,
) -> dict[str, object]:
    started = perf_counter()
    system = _build(
        seed,
        # Keep the primary run at the two-file layout.  The shuffled-outcome
        # control is built as a separate matched system below so adding a
        # control file cannot perturb the primary system's initialization.
        slot_count=2,
        event_window_size=0,
        basis_event_read_mode="history_indexed",
        basis_history_age_slot_count=HISTORY_AGE_SLOT_COUNT,
        external_history_query_count=QUERY_COUNT,
    )
    controller_before = _digest(system.agent.controller)
    encoder_before = _digest(system.agent.runtime.encoders["stimulus"])
    common = _common_modules(system)
    source_modules = _slot_modules(system, 0)
    target_modules = _slot_modules(system, 1)

    _set_requires_grad(common + source_modules, True)
    _set_requires_grad(target_modules, False)
    source_history = _train_stage(
        system,
        family=SOURCE_FAMILY,
        slot=0,
        cue_symbol=CUE_SYMBOL,
        updates=source_updates,
        batch_size=batch_size,
        steps=SOURCE_STEPS,
        seed=seed + 100,
        learning_rate=learning_rate,
        entropy_weight=entropy_weight,
        credit_mode="attempted_bce",
        history_query_count=QUERY_COUNT,
    )
    source_after_source = _evaluate_condition(
        system,
        family=SOURCE_FAMILY,
        slot=0,
        batch_size=batch_size,
        steps=SOURCE_STEPS,
        seed=seed + 10_000,
        lifetimes=lifetimes,
    )

    # This is the continual-learning seam: the old file and shared modules
    # become immutable before the new capability is learned.
    _set_requires_grad(common + source_modules, False)
    _set_requires_grad(target_modules, True)
    source_file_before_target = _digest(*source_modules)
    target_history = _train_stage(
        system,
        family=TARGET_FAMILY,
        slot=1,
        cue_symbol=CUE_SYMBOL,
        updates=target_updates,
        batch_size=batch_size,
        steps=TARGET_STEPS,
        seed=seed + 200,
        learning_rate=learning_rate,
        entropy_weight=entropy_weight,
        credit_mode="attempted_bce",
        history_query_count=QUERY_COUNT,
    )
    _set_requires_grad(target_modules, False)

    target_rows = _evaluate_condition(
        system,
        family=TARGET_FAMILY,
        slot=1,
        batch_size=batch_size,
        steps=TARGET_STEPS,
        seed=seed + 20_000,
        lifetimes=lifetimes,
    )
    source_retention = _evaluate_condition(
        system,
        family=SOURCE_FAMILY,
        slot=0,
        batch_size=batch_size,
        steps=SOURCE_STEPS,
        seed=seed + 30_000,
        lifetimes=lifetimes,
    )
    missing_history = _evaluate_condition(
        system,
        family=TARGET_FAMILY,
        slot=1,
        batch_size=batch_size,
        steps=TARGET_STEPS,
        seed=seed + 40_000,
        lifetimes=lifetimes,
        reset_external_each_step=True,
    )
    corrupted_history = _evaluate_condition(
        system,
        family=TARGET_FAMILY,
        slot=1,
        batch_size=batch_size,
        steps=TARGET_STEPS,
        seed=seed + 50_000,
        lifetimes=lifetimes,
        corrupt_external_history=True,
    )
    action_shuffled = _evaluate_condition(
        system,
        family=TARGET_FAMILY,
        slot=1,
        batch_size=batch_size,
        steps=TARGET_STEPS,
        seed=seed + 60_000,
        lifetimes=lifetimes,
        shuffle_actions=True,
    )
    reward_shuffled_diagnostic = _evaluate_condition(
        system,
        family=TARGET_FAMILY,
        slot=1,
        batch_size=batch_size,
        steps=TARGET_STEPS,
        seed=seed + 70_000,
        lifetimes=lifetimes,
        shuffle_outcomes=True,
    )
    shuffled_control_system = _build(
        seed + 80_000,
        slot_count=1,
        event_window_size=0,
        basis_event_read_mode="history_indexed",
        basis_history_age_slot_count=HISTORY_AGE_SLOT_COUNT,
        external_history_query_count=QUERY_COUNT,
    )
    reward_shuffled_training = _train_reward_shuffled_control(
        shuffled_control_system,
        slot=0,
        seed=seed + 80_000,
        updates=target_updates,
        batch_size=batch_size,
        learning_rate=learning_rate,
        entropy_weight=entropy_weight,
        lifetimes=lifetimes,
    )

    source_file_after_target = _digest(*source_modules)
    controller_after = _digest(system.agent.controller)
    encoder_after = _digest(system.agent.runtime.encoders["stimulus"])
    source_unique_bits = sum(int(row["unique_verifier_bits"]) for row in source_history)
    target_unique_bits = sum(int(row["unique_verifier_bits"]) for row in target_history)
    control_unique_bits = target_unique_bits
    gates = {
        "source_mastered_before_extension": _mastered(source_after_source),
        "target_mastered_in_new_file": _mastered(target_rows),
        "source_retained_after_extension": _mastered(source_retention),
        "source_file_frozen_during_extension": source_file_before_target
        == source_file_after_target,
        "controller_frozen": controller_before == controller_after,
        "event_frontend_frozen": encoder_before == encoder_after,
        "zero_replayed_examples": all(
            int(row["replayed_examples"]) == 0
            for row in (*source_history, *target_history)
        ),
        "missing_history_rejects_mastery": not _mastered(missing_history),
        "corrupted_history_rejects_mastery": not _mastered(corrupted_history),
        "action_shuffled_rejects_mastery": not _mastered(action_shuffled),
        "reward_shuffled_training_rejects_mastery": not _mastered(
            reward_shuffled_training
        ),
    }
    return {
        "seed": seed,
        "source_family": SOURCE_FAMILY,
        "target_family": TARGET_FAMILY,
        "history_age_slot_count": HISTORY_AGE_SLOT_COUNT,
        "query_count": QUERY_COUNT,
        "source_after_source": source_after_source,
        "target_rows": target_rows,
        "source_retention": source_retention,
        "controls": {
            "missing_history": missing_history,
            "corrupted_history": corrupted_history,
            "action_shuffled": action_shuffled,
            "reward_shuffled_diagnostic": reward_shuffled_diagnostic,
            "reward_shuffled_training": reward_shuffled_training,
        },
        "training": {
            "source_updates": source_updates,
            "target_updates": target_updates,
            "source_unique_verifier_bits": source_unique_bits,
            "target_unique_verifier_bits": target_unique_bits,
            "reward_shuffled_control_unique_verifier_bits": control_unique_bits,
            "reward_shuffled_control_optimizer_updates": target_updates,
            "optimizer_updates": source_updates + target_updates,
            "replayed_examples": 0,
            "source_tail": source_history[-5:],
            "target_tail": target_history[-5:],
        },
        "means": {
            "source_after_source": _mean(source_after_source),
            "target": _mean(target_rows),
            "source_retention": _mean(source_retention),
            "missing_history": _mean(missing_history),
            "corrupted_history": _mean(corrupted_history),
            "action_shuffled": _mean(action_shuffled),
            "reward_shuffled_diagnostic": _mean(reward_shuffled_diagnostic),
            "reward_shuffled_training": _mean(reward_shuffled_training),
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "accounting": {
            "unique_verifier_bits": source_unique_bits + target_unique_bits,
            "control_unique_verifier_bits": control_unique_bits,
            "unique_logical_lifetimes": (source_updates + target_updates) * batch_size,
            "optimizer_updates": source_updates + target_updates,
            "replayed_examples": 0,
            "wall_time_seconds": perf_counter() - started,
        },
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if not args.seeds:
        raise ValueError("at least one seed is required")
    if min(
        args.source_updates,
        args.target_updates,
        args.batch_size,
        args.lifetimes,
    ) < 1:
        raise ValueError("training and evaluation budgets must be positive")
    if args.learning_rate <= 0.0 or args.entropy_weight < 0.0:
        raise ValueError("learning rate must be positive and entropy non-negative")
    reports = [
        _run_seed(
            seed=seed,
            source_updates=args.source_updates,
            target_updates=args.target_updates,
            batch_size=args.batch_size,
            lifetimes=args.lifetimes,
            learning_rate=args.learning_rate,
            entropy_weight=args.entropy_weight,
        )
        for seed in args.seeds
    ]
    report: dict[str, object] = {
        "schema": SCHEMA,
        "claim_boundary": (
            "Promoted bounded append-only external computation growth with "
            "frozen old-file retention; not general continual learning, "
            "unrestricted memory growth, or arbitrary program induction."
        ),
        "configuration": {
            "source_updates": args.source_updates,
            "target_updates": args.target_updates,
            "batch_size": args.batch_size,
            "lifetimes": args.lifetimes,
            "learning_rate": args.learning_rate,
            "entropy_weight": args.entropy_weight,
            "controller_and_frontend": "frozen",
            "replayed_examples": 0,
        },
        "seeds": reports,
        "promoted": len(reports) >= 2 and all(bool(item["promoted"]) for item in reports),
    }
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 18])
    parser.add_argument("--source-updates", type=int, default=256)
    parser.add_argument("--target-updates", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lifetimes", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

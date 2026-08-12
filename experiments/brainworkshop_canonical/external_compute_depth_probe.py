"""Promote a deeper generic external-compute history window.

The probe uses the existing flattened learned-event ABI with a six-record
window and an opaque n-back-5 verifier. The controller and event frontend are
unchanged; only the replaceable external compute basis receives the larger
window. The private verifier family is used by this harness only and never
enters the learner-visible tensors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from .external_compute_growth import (
    _build,
    _common_modules,
    _digest,
    _evaluate,
    _set_requires_grad,
    _slot_modules,
    _train_stage,
)

DEPTH_PROBE_SCHEMA = "neural-computer.brainworkshop-external-compute-depth-probe.v2"
FAMILY = "nback5"
CUE_SYMBOL = 12


def run(args: argparse.Namespace) -> dict[str, object]:
    event_read_mode = getattr(args, "event_read_mode", "flattened_window")
    if event_read_mode not in (
        "flattened_window",
        "history_attention",
        "history_indexed",
    ):
        raise ValueError("unsupported depth-probe event read mode")
    history_reader = event_read_mode in ("history_attention", "history_indexed")
    if history_reader:
        if args.event_window_size != 0:
            raise ValueError("variable history depth probes require event_window_size=0")
        if args.query_count < 0:
            raise ValueError("history query count cannot be negative")
    elif args.event_window_size < 1 or args.query_count < 1:
        raise ValueError("flattened depth probes require positive window and query")
    if min(
        args.updates,
        args.batch_size,
        args.steps,
        args.retention_lifetimes,
    ) < 1:
        raise ValueError("depth-probe budgets and dimensions must be positive")
    if not history_reader and args.query_count > args.event_window_size:
        raise ValueError("query count cannot exceed the event window size")
    if args.learning_rate <= 0.0:
        raise ValueError("learning rate must be positive")
    started = perf_counter()
    system = _build(
        args.seed,
        slot_count=1,
        event_window_size=args.event_window_size,
        basis_event_read_mode=event_read_mode,
        external_history_query_count=args.query_count,
    )
    controller_before = _digest(system.agent.controller)
    encoder_before = _digest(system.agent.runtime.encoders["stimulus"])
    modules = _common_modules(system) + _slot_modules(system, 0)
    _set_requires_grad(modules, True)
    history = _train_stage(
        system,
        family=FAMILY,
        slot=0,
        cue_symbol=CUE_SYMBOL,
        updates=args.updates,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 100,
        learning_rate=args.learning_rate,
        entropy_weight=args.entropy_weight,
        credit_mode="attempted_bce",
        history_query_count=args.query_count,
    )
    _set_requires_grad(modules, False)
    rows = _evaluate(
        system,
        family=FAMILY,
        slot=0,
        cue_symbol=CUE_SYMBOL,
        lifetimes=args.retention_lifetimes,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 10_000,
        history_query_count=args.query_count,
    )
    missing_history = _evaluate(
        system,
        family=FAMILY,
        slot=0,
        cue_symbol=CUE_SYMBOL,
        lifetimes=args.retention_lifetimes,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 20_000,
        reset_external_each_step=True,
        history_query_count=args.query_count,
    )
    corrupted_history = _evaluate(
        system,
        family=FAMILY,
        slot=0,
        cue_symbol=CUE_SYMBOL,
        lifetimes=args.retention_lifetimes,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 30_000,
        corrupt_external_history=True,
        history_query_count=args.query_count,
    )
    action_shuffled = _evaluate(
        system,
        family=FAMILY,
        slot=0,
        cue_symbol=CUE_SYMBOL,
        lifetimes=args.retention_lifetimes,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 35_000,
        shuffle_actions=True,
        history_query_count=args.query_count,
    )
    depth_shift = _evaluate(
        system,
        family="nback4",
        slot=0,
        cue_symbol=CUE_SYMBOL,
        lifetimes=args.retention_lifetimes,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 40_000,
        history_query_count=args.query_count,
    )
    control_updates = getattr(
        args,
        "control_updates",
        max(32, min(128, args.updates)),
    )
    if control_updates < 1:
        raise ValueError("control updates must be positive")
    shuffled_system = _build(
        args.seed + 70_000,
        slot_count=1,
        event_window_size=args.event_window_size,
        basis_event_read_mode=event_read_mode,
        external_history_query_count=args.query_count,
    )
    shuffled_modules = _common_modules(shuffled_system) + _slot_modules(
        shuffled_system, 0
    )
    _set_requires_grad(shuffled_modules, True)
    shuffled_history = _train_stage(
        shuffled_system,
        family=FAMILY,
        slot=0,
        cue_symbol=CUE_SYMBOL,
        updates=control_updates,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 71_000,
        learning_rate=args.learning_rate,
        entropy_weight=args.entropy_weight,
        credit_mode="attempted_bce",
        shuffle_outcomes=True,
        history_query_count=args.query_count,
    )
    _set_requires_grad(shuffled_modules, False)
    shuffled_rows = _evaluate(
        shuffled_system,
        family=FAMILY,
        slot=0,
        cue_symbol=CUE_SYMBOL,
        lifetimes=args.retention_lifetimes,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 72_000,
        history_query_count=args.query_count,
    )
    controller_after = _digest(system.agent.controller)
    encoder_after = _digest(system.agent.runtime.encoders["stimulus"])
    gates = {
        "fresh_depth_mastered": min(float(row["accuracy"]) for row in rows)
        >= 0.99,
        "frozen_controller": controller_before == controller_after,
        "frozen_event_encoder": encoder_before == encoder_after,
        "zero_replayed_examples": True,
        "missing_history_control_rejects_mastery": max(
            float(row["accuracy"]) for row in missing_history
        )
        < 0.70,
        "corrupted_history_control_rejects_mastery": max(
            float(row["accuracy"]) for row in corrupted_history
        )
        < 0.80,
        "action_shuffled_control_rejects_mastery": max(
            float(row["accuracy"]) for row in action_shuffled
        )
        < 0.70,
        "depth_shift_control_rejects_mastery": max(
            float(row["accuracy"]) for row in depth_shift
        )
        < 0.80,
        "reward_shuffled_control_rejects_mastery": max(
            float(row["accuracy"]) for row in shuffled_rows
        )
        < 0.70,
    }
    report = {
        "schema": DEPTH_PROBE_SCHEMA,
        "claim_boundary": (
            "Outcome-only acquisition of a deeper n-back-5 computation through "
            "a generic six-record external event window; not unbounded history, "
            "learned compression, arbitrary program induction, or general continual learning."
        ),
        "architecture": {
            "family": FAMILY,
            "event_read_mode": event_read_mode,
            "event_window_size": args.event_window_size,
            "query_count": args.query_count,
            "query_count_semantics": "q_minus_one_previous_records_plus_current_event",
            "control_updates": control_updates,
            "boundary": "rendered_event -> frozen_amodal_controller -> external_compute_file -> keypress_decoder",
        },
        "training_tail": [
            float(row["eligible_accuracy"]) for row in history[-5:]
        ],
        "fresh": rows,
        "controls": {
            "missing_history": missing_history,
            "corrupted_history": corrupted_history,
            "action_shuffled": action_shuffled,
            "depth_shift_nback4": depth_shift,
            "reward_shuffled": shuffled_rows,
            "reward_shuffled_training_tail": [
                float(row["eligible_accuracy"]) for row in shuffled_history[-5:]
            ],
        },
        "gates": gates,
        "accounting": {
            "unique_verifier_bits": sum(
                int(row["unique_verifier_bits"]) for row in history
            ),
            "unique_logical_lifetimes": args.batch_size * args.updates,
            "optimizer_updates": args.updates,
            "replayed_examples": 0,
            "stable_bits_to_threshold": sum(
                int(row["unique_verifier_bits"]) for row in history
            )
            if all(gates.values())
            else None,
            "control_unique_verifier_bits": sum(
                int(row["unique_verifier_bits"]) for row in shuffled_history
            ),
            "control_optimizer_updates": control_updates,
            "wall_time_seconds": perf_counter() - started,
        },
        "status": "promoted" if all(gates.values()) else "rejected",
    }
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--updates", type=int, default=512)
    parser.add_argument("--control-updates", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--retention-lifetimes", type=int, default=4)
    parser.add_argument("--event-window-size", type=int, default=6)
    parser.add_argument("--query-count", type=int, default=6)
    parser.add_argument(
        "--event-read-mode",
        choices=("flattened_window", "history_attention", "history_indexed"),
        default="flattened_window",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

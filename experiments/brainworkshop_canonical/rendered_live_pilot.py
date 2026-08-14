"""Fast rendered vision/audio acquisition audit on the live cognitive tick."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from .rendered_environment import (
    RenderedBrainWorkshopConfig,
    RenderedBrainWorkshopEncoders,
)
from .rendered_live import (
    SourcePreservingTemporalMachine,
    run_rendered_live_lifetime,
)

SCHEMA = "neural-computer.brainworkshop-rendered-live-pilot.v1"
MASTERY_THRESHOLD = 0.80


def _stable_bits(rows: list[dict[str, float | int]]) -> int | None:
    for index, row in enumerate(rows):
        if float(row["heldout_accuracy"]) >= MASTERY_THRESHOLD and all(
            float(later["heldout_accuracy"]) >= MASTERY_THRESHOLD
            for later in rows[index:]
        ):
            return int(row["training_unique_verifier_bits"])
    return None


def _evaluate(
    machine: SourcePreservingTemporalMachine,
    encoders: RenderedBrainWorkshopEncoders,
    *,
    streams: tuple[str, ...],
    steps: int,
    seeds: tuple[int, ...],
    reverse_event_order: bool = False,
    drop_streams: tuple[str, ...] = (),
    reset_history_each_tick: bool = False,
) -> tuple[float, int, int, float]:
    scores: list[float] = []
    bits = 0
    ticks = 0
    p99 = 0.0
    config = RenderedBrainWorkshopConfig(
        n_back=1,
        steps=steps,
        streams=streams,
    )
    for seed in seeds:
        report = run_rendered_live_lifetime(
            machine,
            encoders,
            config,
            seed=seed,
            learn=False,
            sample=False,
            reverse_event_order=reverse_event_order,
            drop_streams=drop_streams,
            reset_history_each_tick=reset_history_each_tick,
        )
        scores.append(report.eligible_accuracy)
        bits += report.unique_verifier_bits
        ticks += report.ticks
        p99 = max(p99, report.total_seconds_p99)
    return sum(scores) / len(scores), bits, ticks, p99


def _condition(
    *,
    streams: tuple[str, ...],
    seed: int,
    train_lifetimes: int,
    train_steps: int,
    evaluation_steps: int,
    evaluation_lifetimes: int,
    checkpoints: tuple[int, ...],
    event_width: int,
    hidden: int,
    learning_rate: float,
) -> tuple[dict[str, object], SourcePreservingTemporalMachine, RenderedBrainWorkshopEncoders]:
    torch.manual_seed(seed)
    encoders = RenderedBrainWorkshopEncoders(event_width, source_key_width=4)
    for parameter in encoders.parameters():
        parameter.requires_grad_(False)
    config = RenderedBrainWorkshopConfig(
        n_back=1,
        steps=train_steps,
        streams=streams,
    )
    machine = SourcePreservingTemporalMachine(
        event_width,
        source_key_width=encoders.source_key_width,
        max_history=4,
        max_sources=2,
        action_count=config.action_count,
        intention_width=event_width,
        hidden=hidden,
        learning_rate=learning_rate,
        sample=True,
    )
    training_bits = 0
    evaluation_bits = 0
    cognitive_ticks = 0
    deadline_misses = 0
    p99 = 0.0
    rows: list[dict[str, float | int]] = []
    for lifetime in range(1, train_lifetimes + 1):
        trained = run_rendered_live_lifetime(
            machine,
            encoders,
            config,
            seed=seed + lifetime,
            learn=True,
            sample=True,
            max_tick_seconds=0.05,
        )
        training_bits += trained.unique_verifier_bits
        cognitive_ticks += trained.ticks
        deadline_misses += trained.deadline_misses
        p99 = max(p99, trained.total_seconds_p99)
        if lifetime not in checkpoints and lifetime != train_lifetimes:
            continue
        heldout, bits, ticks, heldout_p99 = _evaluate(
            machine,
            encoders,
            streams=streams,
            steps=evaluation_steps,
            seeds=tuple(
                seed + 100_000 + lifetime * 100 + index
                for index in range(evaluation_lifetimes)
            ),
        )
        evaluation_bits += bits
        cognitive_ticks += ticks
        p99 = max(p99, heldout_p99)
        rows.append(
            {
                "training_lifetimes": lifetime,
                "training_unique_verifier_bits": training_bits,
                "optimizer_updates": machine.optimizer_updates,
                "heldout_accuracy": heldout,
            }
        )
    return (
        {
            "streams": list(streams),
            "learning_curve": rows,
            "stable_bits_to_threshold": _stable_bits(rows),
            "training_unique_verifier_bits": training_bits,
            "evaluation_unique_verifier_bits": evaluation_bits,
            "training_logical_lifetimes": train_lifetimes,
            "evaluation_logical_lifetimes": len(rows) * evaluation_lifetimes,
            "optimizer_updates": machine.optimizer_updates,
            "replayed_examples": 0,
            "cognitive_ticks": cognitive_ticks,
            "deadline_misses": deadline_misses,
            "maximum_end_to_end_p99_seconds": p99,
        },
        machine,
        encoders,
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    if min(
        args.train_lifetimes,
        args.train_steps,
        args.evaluation_steps,
        args.evaluation_lifetimes,
        args.event_width,
        args.hidden,
    ) < 1:
        raise ValueError("rendered live pilot budgets must be positive")
    if args.train_steps <= 1 or args.evaluation_steps <= 1:
        raise ValueError("rendered live pilot needs target-bearing lifetimes")
    if args.learning_rate <= 0.0:
        raise ValueError("rendered live pilot learning rate must be positive")
    checkpoints = tuple(
        checkpoint
        for checkpoint in args.checkpoints
        if checkpoint <= args.train_lifetimes
    )
    if not checkpoints:
        checkpoints = (args.train_lifetimes,)
    started = perf_counter()
    conditions: list[dict[str, object]] = []
    dual_machine: SourcePreservingTemporalMachine | None = None
    dual_encoders: RenderedBrainWorkshopEncoders | None = None
    for streams in (("vision",), ("audio",), ("vision", "audio")):
        condition, machine, encoders = _condition(
            streams=streams,
            seed=args.seed,
            train_lifetimes=args.train_lifetimes,
            train_steps=args.train_steps,
            evaluation_steps=args.evaluation_steps,
            evaluation_lifetimes=args.evaluation_lifetimes,
            checkpoints=checkpoints,
            event_width=args.event_width,
            hidden=args.hidden,
            learning_rate=args.learning_rate,
        )
        conditions.append(condition)
        if len(streams) == 2:
            dual_machine = machine
            dual_encoders = encoders
    assert dual_machine is not None and dual_encoders is not None
    control_seeds = tuple(args.seed + 900_000 + index for index in range(4))
    control_kwargs = {
        "streams": ("vision", "audio"),
        "steps": args.evaluation_steps,
        "seeds": control_seeds,
    }
    normal, normal_bits, normal_ticks, _ = _evaluate(
        dual_machine,
        dual_encoders,
        **control_kwargs,
    )
    reversed_order, reversed_bits, reversed_ticks, _ = _evaluate(
        dual_machine,
        dual_encoders,
        reverse_event_order=True,
        **control_kwargs,
    )
    no_history, no_history_bits, no_history_ticks, _ = _evaluate(
        dual_machine,
        dual_encoders,
        reset_history_each_tick=True,
        **control_kwargs,
    )
    missing_vision, missing_vision_bits, missing_vision_ticks, _ = _evaluate(
        dual_machine,
        dual_encoders,
        drop_streams=("vision",),
        **control_kwargs,
    )
    missing_audio, missing_audio_bits, missing_audio_ticks, _ = _evaluate(
        dual_machine,
        dual_encoders,
        drop_streams=("audio",),
        **control_kwargs,
    )
    gates = {
        "vision_mechanistic_signal": conditions[0]["stable_bits_to_threshold"]
        is not None,
        "audio_mechanistic_signal": conditions[1]["stable_bits_to_threshold"]
        is not None,
        "dual_mechanistic_signal": conditions[2]["stable_bits_to_threshold"]
        is not None,
        "event_order_invariant": normal == reversed_order,
        "history_is_causal": no_history < MASTERY_THRESHOLD,
        "vision_is_causal": missing_vision < MASTERY_THRESHOLD,
        "audio_is_causal": missing_audio < MASTERY_THRESHOLD,
    }
    report: dict[str, object] = {
        "schema": SCHEMA,
        "status": (
            "mechanistic_rendered_live_signal"
            if all(gates.values())
            else "insufficient_rendered_live_signal"
        ),
        "claim_boundary": (
            "Batch-one live acquisition from clean-room RGB and waveform inputs "
            "with source-preserving temporal memory and exact scalar joint reward; "
            "not physical Brain Workshop control, transfer, or promotion."
        ),
        "configuration": {
            "seed": args.seed,
            "train_lifetimes": args.train_lifetimes,
            "train_steps": args.train_steps,
            "evaluation_steps": args.evaluation_steps,
            "evaluation_lifetimes": args.evaluation_lifetimes,
            "event_width": args.event_width,
            "hidden": args.hidden,
            "batch_size": 1,
            "mastery_threshold": MASTERY_THRESHOLD,
        },
        "conditions": conditions,
        "dual_controls": {
            "normal": normal,
            "reversed_event_order": reversed_order,
            "reset_history_each_tick": no_history,
            "missing_vision": missing_vision,
            "missing_audio": missing_audio,
        },
        "gates": gates,
        "accounting": {
            "training_unique_verifier_bits": sum(
                int(condition["training_unique_verifier_bits"])
                for condition in conditions
            ),
            "evaluation_unique_verifier_bits": sum(
                int(condition["evaluation_unique_verifier_bits"])
                for condition in conditions
            ),
            "control_unique_verifier_bits": sum(
                (
                    normal_bits,
                    reversed_bits,
                    no_history_bits,
                    missing_vision_bits,
                    missing_audio_bits,
                )
            ),
            "training_logical_lifetimes": sum(
                int(condition["training_logical_lifetimes"])
                for condition in conditions
            ),
            "evaluation_logical_lifetimes": sum(
                int(condition["evaluation_logical_lifetimes"])
                for condition in conditions
            ),
            "control_logical_lifetimes": len(control_seeds) * 5,
            "cognitive_ticks": sum(
                int(condition["cognitive_ticks"]) for condition in conditions
            )
            + sum(
                (
                    normal_ticks,
                    reversed_ticks,
                    no_history_ticks,
                    missing_vision_ticks,
                    missing_audio_ticks,
                )
            ),
            "optimizer_updates": sum(
                int(condition["optimizer_updates"]) for condition in conditions
            ),
            "replayed_examples": 0,
            "wall_time_seconds": perf_counter() - started,
            "retention_on_mastered_primitives": "not measured in diagnostic pilot",
            "transfer_ratio_against_fresh_learner": "not measured in diagnostic pilot",
        },
    }
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-out", type=Path)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-lifetimes", type=int, default=40)
    parser.add_argument("--train-steps", type=int, default=24)
    parser.add_argument("--evaluation-steps", type=int, default=64)
    parser.add_argument("--evaluation-lifetimes", type=int, default=4)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[5, 10, 20, 40])
    parser.add_argument("--event-width", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

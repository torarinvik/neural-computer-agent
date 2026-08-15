"""Measure live batch-one n-back acquisition on a fast virtual clock.

This is a diagnostic ladder rung, not a promotion claim. Every optimizer step
is caused by a newly received scalar outcome in the same event-driven runtime
used by a real-clock device. No trajectory is replayed and no correct action or
hidden n-back target crosses the verifier boundary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from .environment import BrainWorkshopEventEncoder
from .live_session import OnlineTemporalCapabilityMachine, run_live_lifetime

SCHEMA = "neural-computer.brainworkshop-live-nback-pilot.v1"


def _stable_bits_to_threshold(
    rows: list[dict[str, float | int]], threshold: float
) -> int | None:
    for index, row in enumerate(rows):
        if float(row["heldout_accuracy"]) >= threshold and all(
            float(later["heldout_accuracy"]) >= threshold
            for later in rows[index:]
        ):
            return int(row["training_unique_verifier_bits"])
    return None


def run(args: argparse.Namespace) -> dict[str, object]:
    if min(
        args.n_back,
        args.train_lifetimes,
        args.train_steps,
        args.evaluate_every,
        args.evaluation_lifetimes,
        args.evaluation_steps,
        args.event_width,
        args.max_history,
    ) < 1:
        raise ValueError("live pilot dimensions and budgets must be positive")
    if args.train_steps <= args.n_back or args.evaluation_steps <= args.n_back:
        raise ValueError("live pilot lifetimes must contain eligible trials")
    if not 0.5 < args.mastery_threshold <= 1.0:
        raise ValueError("mastery threshold must lie in (0.5, 1]")
    started = perf_counter()
    torch.manual_seed(args.seed)
    encoder = BrainWorkshopEventEncoder(4, args.event_width)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    machine = OnlineTemporalCapabilityMachine(
        args.event_width,
        max_history=args.max_history,
        intention_width=args.intention_width,
        hidden=args.hidden,
        learning_rate=args.learning_rate,
        sample=True,
    )
    training_bits = 0
    evaluation_bits = 0
    training_lifetimes = 0
    evaluation_lifetimes = 0
    rows: list[dict[str, float | int]] = []
    p99_seconds: list[float] = []
    total_p99_seconds: list[float] = []
    total_ticks = 0
    deadline_misses = 0
    for lifetime in range(1, args.train_lifetimes + 1):
        trained = run_live_lifetime(
            machine,
            encoder,
            n_back=args.n_back,
            steps=args.train_steps,
            seed=args.seed + lifetime,
            tick_seconds=args.tick_seconds,
            learn=True,
            sample=True,
            max_machine_seconds=args.max_machine_seconds,
        )
        training_bits += trained.unique_verifier_bits
        training_lifetimes += 1
        p99_seconds.append(trained.machine_seconds_p99)
        total_p99_seconds.append(trained.total_seconds_p99)
        total_ticks += trained.ticks
        deadline_misses += trained.deadline_misses
        if lifetime % args.evaluate_every and lifetime != args.train_lifetimes:
            continue
        scores: list[float] = []
        for evaluation in range(args.evaluation_lifetimes):
            heldout = run_live_lifetime(
                machine,
                encoder,
                n_back=args.n_back,
                steps=args.evaluation_steps,
                seed=args.seed + 100_000 + lifetime * 100 + evaluation,
                tick_seconds=args.tick_seconds,
                learn=False,
                sample=False,
                max_machine_seconds=args.max_machine_seconds,
            )
            scores.append(heldout.eligible_accuracy)
            evaluation_bits += heldout.unique_verifier_bits
            evaluation_lifetimes += 1
            p99_seconds.append(heldout.machine_seconds_p99)
            total_p99_seconds.append(heldout.total_seconds_p99)
            total_ticks += heldout.ticks
            deadline_misses += heldout.deadline_misses
        rows.append(
            {
                "training_lifetimes": lifetime,
                "training_unique_verifier_bits": training_bits,
                "optimizer_updates": machine.optimizer_updates,
                "heldout_accuracy": sum(scores) / len(scores),
            }
        )
    stable_bits = _stable_bits_to_threshold(rows, args.mastery_threshold)
    report: dict[str, object] = {
        "schema": SCHEMA,
        "status": (
            "mechanistic_live_signal"
            if stable_bits is not None
            else "insufficient_live_signal"
        ),
        "claim_boundary": (
            "Batch-one immediate outcome updates through the live tick runtime; "
            "not pixel/audio Neural Workshop, autonomous routing, transfer, or "
            "a promoted working-memory capability."
        ),
        "configuration": {
            "n_back": args.n_back,
            "train_lifetimes": args.train_lifetimes,
            "train_steps": args.train_steps,
            "evaluation_lifetimes": args.evaluation_lifetimes,
            "evaluation_steps": args.evaluation_steps,
            "evaluate_every": args.evaluate_every,
            "event_width": args.event_width,
            "max_history": args.max_history,
            "mastery_threshold": args.mastery_threshold,
            "tick_seconds": args.tick_seconds,
            "batch_size": 1,
        },
        "learning_curve": rows,
        "accounting": {
            "training_unique_verifier_bits": training_bits,
            "evaluation_unique_verifier_bits": evaluation_bits,
            "training_logical_lifetimes": training_lifetimes,
            "evaluation_logical_lifetimes": evaluation_lifetimes,
            "optimizer_updates": machine.optimizer_updates,
            "replayed_examples": 0,
            "stable_bits_to_threshold": stable_bits,
            "retention_on_mastered_primitives": "not measured in diagnostic pilot",
            "transfer_ratio_against_fresh_learner": "not measured in diagnostic pilot",
            "maximum_lifetime_tick_p99_seconds": max(p99_seconds, default=0.0),
            "maximum_lifetime_end_to_end_p99_seconds": max(
                total_p99_seconds, default=0.0
            ),
            "cognitive_ticks": total_ticks,
            "deadline_misses": deadline_misses,
            "deadline_miss_fraction": deadline_misses / max(total_ticks, 1),
            "wall_time_seconds": perf_counter() - started,
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
    parser.add_argument("--n-back", type=int, default=1)
    parser.add_argument("--train-lifetimes", type=int, default=20)
    parser.add_argument("--train-steps", type=int, default=24)
    parser.add_argument("--evaluate-every", type=int, default=5)
    parser.add_argument("--evaluation-lifetimes", type=int, default=4)
    parser.add_argument("--evaluation-steps", type=int, default=64)
    parser.add_argument("--event-width", type=int, default=8)
    parser.add_argument("--max-history", type=int, default=4)
    parser.add_argument("--intention-width", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--mastery-threshold", type=float, default=0.80)
    parser.add_argument("--tick-seconds", type=float, default=0.01)
    parser.add_argument("--max-machine-seconds", type=float, default=0.05)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

"""Diagnostic: does longer BPTT through the recurrent core help acquisition?

Every promoted game rung detaches the controller state after each step
(one-step truncated backpropagation), suspected of costing acquisition
quality versus the standalone feedforward slots. This control trains the
identical Snake configuration at several detach intervals with matched
budgets and reports endpoint mastery and acquisition curves. It is a
measurement, not a promotion rung: no gates beyond zero replay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.games_amodal.shared_controller import (
    SharedControllerAgent,
    evaluate_game,
    train_game,
    trainable_parameters,
)


def run(args: argparse.Namespace) -> dict[str, object]:
    eval_seeds = tuple(args.seed + 10_000 + index for index in range(args.eval_seeds))
    results: dict[str, object] = {}
    for interval in args.intervals:
        torch.manual_seed(args.seed)
        agent = SharedControllerAgent(
            event_width=args.event_width,
            intention_width=args.intent_width,
            feedback_width=args.feedback_width,
            hidden=args.hidden,
        )
        history = train_game(
            agent,
            "snake",
            trainable=trainable_parameters(
                [agent.controller, *agent.game_modules("snake")]
            ),
            updates=args.updates,
            batch_size=args.batch_size,
            steps=args.steps,
            seed=args.seed,
            gamma=args.gamma,
            learning_rate=args.learning_rate,
            shuffle_rewards=False,
            detach_interval=interval,
        )
        masteries = [entry["mastery"] for entry in history]
        crossed = [
            entry["update"] for entry in history if entry["mastery"] >= 0.5
        ]
        results[f"interval_{interval}"] = {
            "eval": evaluate_game(
                agent,
                "snake",
                batch_size=args.batch_size,
                steps=args.steps,
                seeds=eval_seeds,
                gamma=args.gamma,
            ),
            "mean_training_mastery": float(sum(masteries) / len(masteries)),
            "updates_to_half_mastery": None if not crossed else float(crossed[0]),
            "no_replay": all(
                entry["replayed_examples"] == 0.0 for entry in history
            ),
        }
    return {
        "seed": args.seed,
        "config": {
            "updates": args.updates,
            "batch_size": args.batch_size,
            "steps": args.steps,
            "intervals": list(args.intervals),
        },
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--intervals", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--updates", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--event-width", type=int, default=64)
    parser.add_argument("--intent-width", type=int, default=32)
    parser.add_argument("--feedback-width", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--eval-seeds", type=int, default=8)
    parser.add_argument("--report-out", type=Path, default=None)
    args = parser.parse_args()
    report = run(args)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()

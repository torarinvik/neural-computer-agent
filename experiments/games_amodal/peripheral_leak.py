"""Diagnostic: how much skill leaks into the per-game peripherals?

The architecture doc requires peripherals to hold only format translation.
This diagnostic measures the violation: after Snake is acquired end to end,
the controller is frozen and Snake's peripherals (frontend, feedback
encoder, decoder) are replaced with fresh random modules that retrain
through the frozen core at several budgets.

* If peripherals held only translation, a small budget recovers full
  mastery through the frozen core (the strategy was in the core).
* The recovery gap at each budget — versus both the original agent and a
  fully-fresh-agent control at the same budget — quantifies the leak.

Measurement only: no promotion gates beyond zero replay.
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


def fresh_peripherals_like(
    agent: SharedControllerAgent, game: str, *, seed: int
) -> None:
    """Replace one game's peripherals with fresh random modules in place."""

    torch.manual_seed(seed)
    donor = SharedControllerAgent(
        event_width=agent.controller.width,
        intention_width=agent.controller.intention_width,
        feedback_width=agent.controller.feedback_width,
        hidden=agent.runtime.output_bus.decoders[game].hidden,
        games=agent.games,
    )
    agent.runtime.encoders[game].load_state_dict(
        donor.runtime.encoders[game].state_dict()
    )
    agent.runtime.output_bus.decoders[game].load_state_dict(
        donor.runtime.output_bus.decoders[game].state_dict()
    )
    agent.feedback_encoders[game].load_state_dict(
        donor.feedback_encoders[game].state_dict()
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(args.seed)
    build = {
        "event_width": args.event_width,
        "intention_width": args.intent_width,
        "feedback_width": args.feedback_width,
        "hidden": args.hidden,
    }
    eval_seeds = tuple(args.seed + 10_000 + index for index in range(args.eval_seeds))
    eval_common = {
        "batch_size": args.batch_size,
        "steps": args.steps,
        "seeds": eval_seeds,
        "gamma": args.gamma,
    }
    train_common = {
        "batch_size": args.batch_size,
        "steps": args.steps,
        "gamma": args.gamma,
        "learning_rate": args.learning_rate,
        "shuffle_rewards": False,
    }

    agent = SharedControllerAgent(**build)
    train_game(
        agent,
        "snake",
        trainable=trainable_parameters(
            [agent.controller, *agent.game_modules("snake")]
        ),
        updates=args.updates,
        seed=args.seed,
        **train_common,
    )
    original = evaluate_game(agent, "snake", **eval_common)
    checkpoint = {
        key: value.detach().clone()
        for key, value in agent.state_dict().items()
    }

    recovery: dict[str, object] = {}
    for budget in args.recovery_budgets:
        swapped = SharedControllerAgent(**build)
        swapped.load_state_dict(checkpoint)
        fresh_peripherals_like(swapped, "snake", seed=args.seed + 70_000)
        for parameter in swapped.controller.parameters():
            parameter.requires_grad_(False)
        history = train_game(
            swapped,
            "snake",
            trainable=trainable_parameters(swapped.game_modules("snake")),
            updates=budget,
            seed=args.seed + 80_000,
            **train_common,
        )
        fresh_control = SharedControllerAgent(**build)
        for parameter in fresh_control.controller.parameters():
            parameter.requires_grad_(False)
        control_history = train_game(
            fresh_control,
            "snake",
            trainable=trainable_parameters(fresh_control.game_modules("snake")),
            updates=budget,
            seed=args.seed + 80_000,
            **train_common,
        )
        recovery[f"budget_{budget}"] = {
            "through_trained_frozen_core": evaluate_game(
                swapped, "snake", **eval_common
            ),
            "through_random_frozen_core": evaluate_game(
                fresh_control, "snake", **eval_common
            ),
            "no_replay": all(
                entry["replayed_examples"] == 0.0
                for entry in history + control_history
            ),
        }
    return {
        "seed": args.seed,
        "config": {
            "updates": args.updates,
            "recovery_budgets": list(args.recovery_budgets),
            "batch_size": args.batch_size,
            "steps": args.steps,
        },
        "original_mastery": original,
        "recovery": recovery,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=600)
    parser.add_argument(
        "--recovery-budgets", type=int, nargs="+", default=[100, 300]
    )
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

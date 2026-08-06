"""Second-game growth rung: acquire Pong with all Snake state frozen.

This is the first cross-game continual-learning audit for this repository.
The rung first reproduces the promoted Snake acquisition, records its
retention audit, and hashes every Snake parameter.  It then freezes the whole
Snake path and trains an isolated Pong slot (its own frontend, adapter, and
decoder) from scalar outcomes only.  After Pong acquisition it re-audits
Snake on the identical evaluation seeds and verifies the parameter hash is
bit-for-bit unchanged.

Hard gates:

* Snake acquired, then re-audited unchanged after Pong training (mastery and
  parameter hash),
* Pong acquired above the mastery threshold,
* Pong random and reward-shuffled clones remain near chance,
* both capabilities protected in one retention ledger,
* zero replay anywhere.

This promotes isolated-slot growth only; learned routing between games from
opaque events alone remains the next rung.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from experiments.games_amodal.environments import PongVerifier, SnakeVerifier
from experiments.games_amodal.snake_acquisition import (
    SnakePolicy,
    capability_key,
    evaluate,
    train_reward_only,
)
from neural_computer import CapabilityRetentionLedger, RetentionPolicyConfig


def parameter_digest(policy: SnakePolicy) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(policy.state_dict().items()):
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(args.seed)
    shape = {
        "height": 8,
        "width": 8,
        "event_width": args.event_width,
        "intent_width": args.intent_width,
        "hidden": args.hidden,
    }
    eval_seeds = tuple(args.seed + 10_000 + index for index in range(args.eval_seeds))
    common = {
        "batch_size": args.batch_size,
        "steps": args.steps,
        "gamma": args.gamma,
    }

    snake = SnakePolicy(channels=3, action_count=4, **shape)
    train_reward_only(
        snake,
        updates=args.updates,
        seed=args.seed,
        learning_rate=args.learning_rate,
        shuffle_rewards=False,
        verifier_factory=SnakeVerifier,
        **common,
    )
    snake_before = evaluate(
        snake, seeds=eval_seeds, verifier_factory=SnakeVerifier, **common
    )
    digest_before = parameter_digest(snake)

    for parameter in snake.parameters():
        parameter.requires_grad_(False)

    pong = SnakePolicy(channels=2, action_count=3, **shape)
    random_clone = SnakePolicy(channels=2, action_count=3, **shape)
    shuffled_clone = SnakePolicy(channels=2, action_count=3, **shape)
    shuffled_clone.load_state_dict(pong.state_dict())
    history = train_reward_only(
        pong,
        updates=args.updates,
        seed=args.seed + 50_000,
        learning_rate=args.learning_rate,
        shuffle_rewards=False,
        verifier_factory=PongVerifier,
        **common,
    )
    shuffled_history = train_reward_only(
        shuffled_clone,
        updates=args.updates,
        seed=args.seed + 50_000,
        learning_rate=args.learning_rate,
        shuffle_rewards=True,
        verifier_factory=PongVerifier,
        **common,
    )

    pong_conditions = {
        "trained": evaluate(
            pong, seeds=eval_seeds, verifier_factory=PongVerifier, **common
        ),
        "random": evaluate(
            random_clone, seeds=eval_seeds, verifier_factory=PongVerifier, **common
        ),
        "shuffled": evaluate(
            shuffled_clone, seeds=eval_seeds, verifier_factory=PongVerifier, **common
        ),
    }
    snake_after = evaluate(
        snake, seeds=eval_seeds, verifier_factory=SnakeVerifier, **common
    )
    digest_after = parameter_digest(snake)

    ledger = CapabilityRetentionLedger(
        args.event_width, config=RetentionPolicyConfig()
    )
    snake_key = capability_key(
        snake,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 20_000,
        verifier_factory=SnakeVerifier,
    )
    pong_key = capability_key(
        pong,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 20_000,
        verifier_factory=PongVerifier,
    )
    for score in snake_after["per_seed_mastery"]:
        ledger.observe(snake_key, float(score))
    for score in pong_conditions["trained"]["per_seed_mastery"]:
        ledger.observe(pong_key, float(score))

    snake_mastery_before = float(snake_before["mastery"])
    snake_mastery_after = float(snake_after["mastery"])
    pong_mastery = float(pong_conditions["trained"]["mastery"])
    gates = {
        "snake_acquired": snake_mastery_before >= args.mastery_gate,
        "snake_retained": snake_mastery_after >= args.mastery_gate
        and abs(snake_mastery_after - snake_mastery_before) < 1e-9,
        "snake_state_unchanged": digest_after == digest_before,
        "pong_acquired": pong_mastery >= args.mastery_gate,
        "pong_random_near_chance": float(pong_conditions["random"]["mastery"])
        <= args.null_gate,
        "pong_shuffled_near_chance": float(pong_conditions["shuffled"]["mastery"])
        <= args.null_gate,
        "both_capabilities_protected": ledger.is_protected(snake_key)
        and ledger.is_protected(pong_key),
        "no_replay": all(
            entry["replayed_examples"] == 0.0
            for entry in history + shuffled_history
        ),
    }
    return {
        "seed": args.seed,
        "config": {
            "updates": args.updates,
            "batch_size": args.batch_size,
            "steps": args.steps,
            "gamma": args.gamma,
            "learning_rate": args.learning_rate,
            "event_width": args.event_width,
            "intent_width": args.intent_width,
            "hidden": args.hidden,
            "eval_seeds": args.eval_seeds,
        },
        "snake": {
            "before": snake_before,
            "after": snake_after,
            "digest_before": digest_before,
            "digest_after": digest_after,
        },
        "pong": pong_conditions,
        "gates": gates,
        "promoted": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--event-width", type=int, default=64)
    parser.add_argument("--intent-width", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--eval-seeds", type=int, default=8)
    parser.add_argument("--mastery-gate", type=float, default=0.8)
    parser.add_argument("--null-gate", type=float, default=0.2)
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

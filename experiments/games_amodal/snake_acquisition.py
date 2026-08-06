"""First-game acquisition rung: learn Snake from scalar outcomes only.

The verifier keeps all game state and reward rules private.  The learner is a
caller-owned frontend encoder, an intent adapter, and a ``KeypressDecoder``
with exact propensity accounting.  Every update runs fresh verifier lifetimes
and no trajectory or verifier state is ever replayed into a later loss.

The audit reports three conditions on identical evaluation seeds:

* ``trained``   -- the reward-trained policy, evaluated greedily,
* ``random``    -- an untrained clone (chance baseline),
* ``shuffled``  -- a clone trained with batch-shuffled rewards (causal null).

The mastery score for one lifetime row is ``1`` when its total private reward
is positive, so chance sits near zero.  A deliberate post-acquisition audit
writes the ledger; training and controls never touch retention state.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from experiments.games_amodal.environments import SnakeVerifier
from experiments.games_amodal.train import GridEventEncoder
from neural_computer import (
    CapabilityRetentionLedger,
    KeypressDecoder,
    RetentionPolicyConfig,
)


@dataclass(frozen=True)
class RolloutSummary:
    """Outcome-only rollout statistics for one batch of lifetimes."""

    total_reward: torch.Tensor
    foods: torch.Tensor
    survival_steps: torch.Tensor
    advantage: torch.Tensor | None
    log_propensity: torch.Tensor | None
    mask: torch.Tensor | None

    @property
    def mastery(self) -> float:
        return float((self.total_reward > 0).float().mean())


class SnakePolicy(nn.Module):
    """Frontend encoder, intent adapter, and keypress decoder for Snake."""

    def __init__(
        self,
        *,
        height: int,
        width: int,
        event_width: int,
        intent_width: int,
        hidden: int,
    ) -> None:
        super().__init__()
        self.encoder = GridEventEncoder(
            channels=3, height=height, width=width, event_width=event_width
        )
        self.intent_adapter = nn.Sequential(
            nn.Linear(event_width, hidden),
            nn.GELU(),
            nn.Linear(hidden, intent_width),
        )
        self.decoder = KeypressDecoder(intent_width, 4, hidden=hidden)

    def decide(self, observation: torch.Tensor, *, sample: bool):
        event = self.encoder(observation)
        intention = self.intent_adapter(event.payload)
        return event, self.decoder.decide(intention, sample=sample)


def rollout(
    policy: SnakePolicy,
    verifier: SnakeVerifier,
    *,
    steps: int,
    seed: int,
    sample: bool,
    gamma: float,
) -> RolloutSummary:
    verifier.reset(seed=seed)
    batch = verifier.batch_size
    rewards: list[torch.Tensor] = []
    log_propensities: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    alive = torch.ones(batch, dtype=torch.bool)
    for _ in range(steps):
        if not bool(alive.any()):
            break
        masks.append(alive.float())
        _, decision = policy.decide(verifier.observation(), sample=sample)
        log_propensities.append(decision.propensity.clamp_min(1e-8).log())
        outcome = verifier.step(decision.key_index)
        rewards.append(outcome.reward)
        alive = outcome.alive
    reward_matrix = torch.stack(rewards, dim=1)
    mask_matrix = torch.stack(masks, dim=1)
    returns = torch.zeros_like(reward_matrix)
    running = torch.zeros(batch)
    for position in range(reward_matrix.shape[1] - 1, -1, -1):
        running = reward_matrix[:, position] + gamma * running
        returns[:, position] = running
    advantage = None
    propensity_matrix = None
    if sample:
        advantage = returns.detach()
        advantage = advantage - (
            (advantage * mask_matrix).sum() / mask_matrix.sum().clamp_min(1.0)
        )
        propensity_matrix = torch.stack(log_propensities, dim=1)
    return RolloutSummary(
        total_reward=reward_matrix.sum(dim=1),
        foods=(reward_matrix > 0).float().sum(dim=1),
        survival_steps=mask_matrix.sum(dim=1),
        advantage=advantage,
        log_propensity=propensity_matrix,
        mask=mask_matrix if sample else None,
    )


def train_reward_only(
    policy: SnakePolicy,
    *,
    updates: int,
    batch_size: int,
    steps: int,
    seed: int,
    gamma: float,
    learning_rate: float,
    shuffle_rewards: bool,
) -> list[dict[str, float]]:
    """Train from fresh scalar outcomes only; no lifetime is ever replayed.

    ``shuffle_rewards`` permutes each update's realized returns across the
    batch before the gradient step, destroying the action-outcome link while
    preserving the reward marginal.  It is the causal null control.
    """

    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []
    for update in range(updates):
        verifier = SnakeVerifier(batch_size=batch_size, seed=seed + update)
        summary = rollout(
            policy,
            verifier,
            steps=steps,
            seed=seed + update,
            sample=True,
            gamma=gamma,
        )
        assert summary.advantage is not None
        assert summary.log_propensity is not None
        assert summary.mask is not None
        advantage = summary.advantage
        if shuffle_rewards:
            advantage = advantage[torch.randperm(advantage.shape[0])]
        loss_terms = advantage * summary.log_propensity * summary.mask
        loss = -loss_terms.sum() / loss_terms.shape[0]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()
        history.append(
            {
                "update": float(update + 1),
                "loss": float(loss.detach()),
                "mastery": summary.mastery,
                "mean_foods": float(summary.foods.mean()),
                "mean_survival": float(summary.survival_steps.mean()),
                "replayed_examples": 0.0,
            }
        )
    return history


def evaluate(
    policy: SnakePolicy,
    *,
    batch_size: int,
    steps: int,
    seeds: tuple[int, ...],
    gamma: float,
) -> dict[str, object]:
    masteries: list[float] = []
    foods: list[float] = []
    survival: list[float] = []
    for seed in seeds:
        verifier = SnakeVerifier(batch_size=batch_size, seed=seed)
        with torch.no_grad():
            summary = rollout(
                policy,
                verifier,
                steps=steps,
                seed=seed,
                sample=False,
                gamma=gamma,
            )
        masteries.append(summary.mastery)
        foods.append(float(summary.foods.mean()))
        survival.append(float(summary.survival_steps.mean()))
    return {
        "per_seed_mastery": masteries,
        "mastery": float(torch.tensor(masteries).mean()),
        "mean_foods": float(torch.tensor(foods).mean()),
        "mean_survival": float(torch.tensor(survival).mean()),
    }


def capability_key(
    policy: SnakePolicy, *, batch_size: int, steps: int, seed: int
) -> torch.Tensor:
    """Derive an opaque episodic context key from fresh greedy events."""

    verifier = SnakeVerifier(batch_size=batch_size, seed=seed)
    verifier.reset(seed=seed)
    payloads: list[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(steps):
            event, decision = policy.decide(verifier.observation(), sample=False)
            payloads.append(event.payload.mean(dim=0))
            outcome = verifier.step(decision.key_index)
            if not bool(outcome.alive.any()):
                break
    return F.normalize(torch.stack(payloads).mean(dim=0), dim=-1)


def run(args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(args.seed)
    build = {
        "height": 8,
        "width": 8,
        "event_width": args.event_width,
        "intent_width": args.intent_width,
        "hidden": args.hidden,
    }
    trained = SnakePolicy(**build)
    random_clone = SnakePolicy(**build)
    shuffled_clone = SnakePolicy(**build)
    shuffled_clone.load_state_dict(trained.state_dict())

    history = train_reward_only(
        trained,
        updates=args.updates,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        shuffle_rewards=False,
    )
    shuffled_history = train_reward_only(
        shuffled_clone,
        updates=args.updates,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        shuffle_rewards=True,
    )

    eval_seeds = tuple(args.seed + 10_000 + index for index in range(args.eval_seeds))
    conditions = {
        "trained": evaluate(
            trained,
            batch_size=args.batch_size,
            steps=args.steps,
            seeds=eval_seeds,
            gamma=args.gamma,
        ),
        "random": evaluate(
            random_clone,
            batch_size=args.batch_size,
            steps=args.steps,
            seeds=eval_seeds,
            gamma=args.gamma,
        ),
        "shuffled": evaluate(
            shuffled_clone,
            batch_size=args.batch_size,
            steps=args.steps,
            seeds=eval_seeds,
            gamma=args.gamma,
        ),
    }

    ledger = CapabilityRetentionLedger(
        args.event_width, config=RetentionPolicyConfig()
    )
    key = capability_key(
        trained, batch_size=args.batch_size, steps=args.steps, seed=args.seed + 20_000
    )
    for score in conditions["trained"]["per_seed_mastery"]:
        status = ledger.observe(key, float(score))
    protected = ledger.is_protected(key)

    mastery = float(conditions["trained"]["mastery"])
    random_mastery = float(conditions["random"]["mastery"])
    shuffled_mastery = float(conditions["shuffled"]["mastery"])
    gates = {
        "acquired": mastery >= args.mastery_gate,
        "random_near_chance": random_mastery <= args.null_gate,
        "shuffled_near_chance": shuffled_mastery <= args.null_gate,
        "capability_protected": bool(protected),
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
        "conditions": conditions,
        "training_tail": history[-5:],
        "retention": {
            "protected": bool(protected),
            "observations": status.observations,
        },
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

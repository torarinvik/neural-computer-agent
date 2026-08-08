"""Solo ceilings for the battery at the fast-iteration budget.

Each battery game trained alone, no bank, small budget. The resulting
per-game mastery is the denominator for every bank claim at this budget:
a bank score is judged against what the plant can do with the whole
model to itself, not against 1.0.
"""

from __future__ import annotations

import argparse
import json

import torch

from experiments.games_amodal.fragment_bank import (
    NoveltyBonus,
    OutcomeNovelty,
    ValueHead,
    battery_suite,
    mastery,
    rollout_family,
)
from experiments.games_amodal.shared_controller import (
    SharedControllerAgent,
    trainable_parameters,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=69316)
    parser.add_argument("--variant", type=str, required=True)
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--egocentric", action="store_true")
    parser.add_argument("--per-step-baseline", action="store_true")
    parser.add_argument("--normalize-advantage", action="store_true")
    parser.add_argument("--critic", action="store_true")
    parser.add_argument("--novelty", type=float, default=0.0)
    parser.add_argument("--outcome-novelty", type=float, default=0.0)
    parser.add_argument("--gae", type=float, default=-1.0)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--critic-weight", type=float, default=0.5)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--event-width", type=int, default=64)
    parser.add_argument("--entropy", type=float, default=0.0)
    parser.add_argument("--egocentric-crop", dest="egocentric", action="store_const", const="crop")
    parser.add_argument("--conv-screen", action="store_true")
    args = parser.parse_args()

    from experiments.games_amodal.game_family import FamilyConfig
    extras = [
        FamilyConfig(navigate=True, name="navigate1"),
        FamilyConfig(intercept=1, name="intercept1"),
        FamilyConfig(forage=1, name="forageA"),
        FamilyConfig(collect=1, name="collect1"),
    ]
    from experiments.games_amodal.fragment_bank import compose_suite
    ctrain, choldout = compose_suite()
    extras = extras + ctrain + choldout
    train, holdout = battery_suite()
    config = next(
        v for v in train + holdout + extras if v.name == args.variant
    )
    torch.manual_seed(args.seed)
    agent = SharedControllerAgent(
        event_width=args.event_width,
        intention_width=32,
        feedback_width=16,
        hidden=args.hidden,
        event_window_capacity=8,
        shared_drivers=True,
        conv_screen=args.conv_screen,
    )
    critic = ValueHead(intention_width=32) if args.critic else None
    novelty = (
        NoveltyBonus(channels=3, height=8, width=8)
        if args.novelty > 0.0 else None
    )
    outcome = (
        OutcomeNovelty() if args.outcome_novelty > 0.0 else None
    )
    params = trainable_parameters(
        [agent.controller, *agent.game_modules(agent.games[0])]
    )
    if critic is not None:
        params = list(params) + list(critic.parameters())
    if novelty is not None:
        params = list(params) + list(novelty.predictor.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)
    for update in range(args.updates):
        summary = rollout_family(
            agent,
            config,
            None,
            batch_size=args.batch_size,
            steps=args.steps,
            seed=args.seed + update,
            sample=True,
            gamma=args.gamma,
            egocentric=args.egocentric,
            per_step_baseline=args.per_step_baseline,
            normalize_advantage=args.normalize_advantage,
            critic=critic,
            novelty=novelty,
            novelty_weight=args.novelty,
            outcome_novelty=outcome,
            outcome_weight=args.outcome_novelty,
            gae_lambda=args.gae,
        )
        terms = summary["advantage"] * summary["log_propensity"] * summary["mask"]
        loss = -terms.sum() / terms.shape[0]
        if novelty is not None:
            # Train the predictor toward the frozen target: novelty decays
            # as the agent revisits a state, so the pressure keeps moving.
            loss = loss + (
                summary["novelty"] * summary["mask"]
            ).sum() / summary["mask"].sum().clamp_min(1.0)
        if critic is not None:
            # Fit the critic to the returns it is explaining away.
            error = (summary["value"] - summary["returns"].detach()).square()
            loss = loss + args.critic_weight * (
                error * summary["mask"]
            ).sum() / summary["mask"].sum().clamp_min(1.0)
        if args.entropy > 0.0:
            # Entropy bonus: a policy that collapses early to a wrong
            # deterministic action stops sampling the evidence that would
            # correct it -- the classic seed-lottery mechanism.
            logp = torch.log_softmax(summary["logits"], dim=-1)
            entropy = -(logp.exp() * logp).sum(dim=-1)
            loss = loss - args.entropy * (
                entropy * summary["mask"]
            ).sum() / summary["mask"].sum().clamp_min(1.0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

    scores = []
    for index in range(4):
        with torch.no_grad():
            summary = rollout_family(
                agent,
                config,
                None,
                batch_size=args.batch_size,
                steps=args.steps,
                seed=args.seed + 10_000 + index,
                sample=False,
                gamma=args.gamma,
                egocentric=args.egocentric,
            )
        scores.append(mastery(summary, config))
    print(
        json.dumps(
            {
                "variant": config.name,
                "seed": args.seed,
                "updates": args.updates,
                "solo_ceiling": float(torch.tensor(scores).mean()),
            }
        )
    )


if __name__ == "__main__":
    main()

"""Is our consolidation anchor real, or noise rescaled to look confident?

Literature prompt: the empirical Fisher of a policy-gradient agent is built
from score-function gradients, which vanish as the policy saturates. We
normalise each game's Fisher to unit mean, so the *magnitude* collapse is
divided out -- but normalisation cannot restore signal-to-noise. If a
mastered game's Fisher is mostly sampling noise, unit-mean normalisation
amplifies that noise into a confident-looking protection pattern, and
arbitrated consolidation then protects arbitrary directions.

Test: train one game, and at checkpoints estimate the Fisher TWICE from
independent rollout seeds. Correlate the two estimates.

  high correlation  -> the protection pattern is a real property of the task
  correlation -> 0  -> we are protecting noise, and more mastery makes it worse

Reported against policy entropy, which is the quantity the mechanism
predicts drives it.
"""

from __future__ import annotations

import argparse
import json

import torch

from experiments.games_amodal.fragment_bank import (
    battery_suite,
    mastery,
    rollout_family,
)
from experiments.games_amodal.shared_controller import (
    SharedControllerAgent,
    trainable_parameters,
)
from experiments.games_amodal.two_speed_battery import (
    family_fisher,
    plant_named_parameters,
)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=69316)
parser.add_argument("--variant", type=str, default="choiceA")
parser.add_argument("--updates", type=int, default=400)
parser.add_argument("--every", type=int, default=50)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--steps", type=int, default=24)
parser.add_argument("--gamma", type=float, default=0.95)
parser.add_argument("--fisher-batches", type=int, default=4)
parser.add_argument("--fisher-temperature", type=float, default=1.0)
parser.add_argument("--egocentric", type=str, default="")
args = parser.parse_args()

train, holdout = battery_suite()
config = next(v for v in train + holdout if v.name == args.variant)

torch.manual_seed(args.seed)
agent = SharedControllerAgent(
    event_width=64, intention_width=32, feedback_width=16, hidden=32,
    event_window_capacity=8, shared_drivers=True,
)
params = trainable_parameters(
    [agent.controller, *agent.game_modules(agent.games[0])]
)
optimizer = torch.optim.Adam(params, lr=1e-3)
named = plant_named_parameters(agent, include_screen=True)


def flat(fisher: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([fisher[name].reshape(-1) for name, _ in named])


def probe(update: int) -> dict[str, float]:
    """Two independent Fisher estimates, plus entropy and mastery."""
    with torch.no_grad():
        summary = rollout_family(
            agent, config, None, batch_size=args.batch_size,
            steps=args.steps, seed=args.seed + 500_000, sample=True,
            gamma=args.gamma, egocentric=args.egocentric,
        )
    mask = summary["mask"]
    # E[-log p(a)] under a ~ p is an unbiased entropy estimate.
    entropy = float(
        (-summary["log_propensity"] * mask).sum() / mask.sum().clamp_min(1.0)
    )
    score = mastery(
        {"total_reward": summary["total_reward"], "mask": mask}, config
    )
    first = flat(family_fisher(
        agent, config, None, named, args=args, seed=args.seed + 100_000 + update
    ))
    second = flat(family_fisher(
        agent, config, None, named, args=args, seed=args.seed + 300_000 + update
    ))
    # Pearson over the per-parameter importance vector: does an independent
    # estimate of "what matters" agree with this one?
    centred_a = first - first.mean()
    centred_b = second - second.mean()
    correlation = float(
        (centred_a * centred_b).sum()
        / (centred_a.norm() * centred_b.norm()).clamp_min(1e-12)
    )
    # Where the penalty actually bites: the top-1% most-protected params.
    cut = int(first.numel() * 0.01)
    top_a = set(torch.topk(first, cut).indices.tolist())
    top_b = set(torch.topk(second, cut).indices.tolist())
    overlap = len(top_a & top_b) / max(cut, 1)
    return {
        "update": update, "mastery": float(score), "entropy": entropy,
        "fisher_correlation": correlation, "top1pct_overlap": overlap,
    }


rows = [probe(0)]
for update in range(args.updates):
    summary = rollout_family(
        agent, config, None, batch_size=args.batch_size, steps=args.steps,
        seed=args.seed + update, sample=True, gamma=args.gamma,
        egocentric=args.egocentric,
    )
    terms = summary["advantage"] * summary["log_propensity"] * summary["mask"]
    loss = -terms.sum() / terms.shape[0]
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    optimizer.step()
    if (update + 1) % args.every == 0:
        rows.append(probe(update + 1))

print(json.dumps({"seed": args.seed, "variant": args.variant, "rows": rows}))

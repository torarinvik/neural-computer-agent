"""Is our winner-take-all actually gradient conflict?

We have spent a lot of the program treating twin winner-take-all as a
routing/balancing problem: diversity penalties, laggard-preferential
sampling, uniform floors. The multi-task literature says to check
something cheaper first. PCGrad's "tragic triad" holds that joint
training is harmed when task gradients CONFLICT (negative cosine) and
DOMINATE (large magnitude imbalance). If our two contexts have positive
cosine, then no amount of gradient surgery or routing machinery can
help, and the failure is an acquisition/sampling problem instead.
PopArt makes the complementary point: the usual cause of one task
swamping another is reward scale and density, not representational
conflict.

Both are one measurement. Per update, take each context's gradient on
the shared plant separately and report:

  cosine        cos(g_A, g_B) over the flattened plant gradient
  norm_ratio    |g_A| / |g_B|, the domination term
  return_ratio  mean |return| per context, the PopArt term

Interpretation:
  cosine < 0                  -> genuine conflict; gradient surgery is on-topic
  cosine > 0, norms lopsided  -> domination; per-context normalisation is the fix
  cosine > 0, norms matched   -> neither; the problem is acquisition, and the
                                 routing work was aimed at the wrong thing
"""

from __future__ import annotations

import argparse
import json

import torch

from experiments.games_amodal.fragment_bank import (
    FragmentBank,
    battery_suite,
    mastery,
    rollout_family,
)
from experiments.games_amodal.shared_controller import (
    SharedControllerAgent,
    trainable_parameters,
)
from experiments.games_amodal.two_speed_battery import plant_named_parameters

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=69316)
parser.add_argument("--updates", type=int, default=400)
parser.add_argument("--every", type=int, default=25)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--steps", type=int, default=24)
parser.add_argument("--gamma", type=float, default=0.95)
parser.add_argument(
    "--bank", action="store_true",
    help="give each twin its own oracle fragments (the setting that matters)",
)
parser.add_argument("--fragments", type=int, default=6)
parser.add_argument("--per-variant", type=int, default=2)
parser.add_argument("--tokens", type=int, default=2)
args = parser.parse_args()

train, holdout = battery_suite()
twins = [
    v for v in train + holdout if v.name in ("choiceA", "choiceB")
]
assert len(twins) == 2, [v.name for v in twins]

torch.manual_seed(args.seed)
agent = SharedControllerAgent(
    event_width=64, intention_width=32, feedback_width=16, hidden=32,
    event_window_capacity=8, shared_drivers=True,
)
params = trainable_parameters(
    [agent.controller, *agent.game_modules(agent.games[0])]
)
bank = None
if args.bank:
    bank = FragmentBank(
        fragments=args.fragments, tokens_per_fragment=args.tokens,
        width=64, variants=[v.name for v in twins],
    )
    params = list(params) + list(bank.parameters())
optimizer = torch.optim.Adam(params, lr=1e-3)
named = plant_named_parameters(agent, include_screen=True)


def fragments_for(config):
    """This twin's own oracle fragments — disjoint from the other's."""
    if bank is None:
        return None
    return bank.fetch(bank.oracle_indices(config.name, args.per_variant))


def context_gradient(config, seed: int):
    """REINFORCE gradient on the shared plant for one context alone."""
    summary = rollout_family(
        agent, config, fragments_for(config), batch_size=args.batch_size,
        steps=args.steps, seed=seed, sample=True, gamma=args.gamma,
    )
    terms = (
        summary["advantage"] * summary["log_propensity"] * summary["mask"]
    )
    loss = -terms.sum() / terms.shape[0]
    agent.zero_grad(set_to_none=True)
    loss.backward()
    flat = torch.cat([
        (parameter.grad if parameter.grad is not None
         else torch.zeros_like(parameter)).reshape(-1)
        for _, parameter in named
    ]).clone()
    score = mastery(
        {"total_reward": summary["total_reward"], "mask": summary["mask"]},
        config,
    )
    magnitude = float(summary["returns"].abs().mean()) if (
        summary.get("returns") is not None
    ) else float(summary["total_reward"].abs().mean())
    agent.zero_grad(set_to_none=True)
    return flat, float(score), magnitude


rows = []
for update in range(args.updates):
    if update % args.every == 0:
        grad_a, score_a, ret_a = context_gradient(twins[0], args.seed + update)
        grad_b, score_b, ret_b = context_gradient(twins[1], args.seed + update)
        cosine = float(
            (grad_a * grad_b).sum()
            / (grad_a.norm() * grad_b.norm()).clamp_min(1e-12)
        )
        rows.append({
            "update": update,
            "cosine": cosine,
            "norm_a": float(grad_a.norm()),
            "norm_b": float(grad_b.norm()),
            "norm_ratio": float(
                grad_a.norm() / grad_b.norm().clamp_min(1e-12)
            ),
            "mastery_a": score_a, "mastery_b": score_b,
            "return_a": ret_a, "return_b": ret_b,
        })
    # Alternate the contexts so both actually train, as the battery does.
    config = twins[update % 2]
    summary = rollout_family(
        agent, config, fragments_for(config), batch_size=args.batch_size,
        steps=args.steps, seed=args.seed + 900_000 + update, sample=True,
        gamma=args.gamma,
    )
    terms = summary["advantage"] * summary["log_propensity"] * summary["mask"]
    loss = -terms.sum() / terms.shape[0]
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    optimizer.step()

print(json.dumps({"seed": args.seed, "bank": bool(args.bank), "rows": rows}))

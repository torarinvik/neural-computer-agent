"""Measured chance floor for every battery game.

SOLO_CEILINGS gives the denominator for a bank claim: what the plant
achieves with the whole model to itself. The complement was never
measured -- the FLOOR, i.e. what a policy carrying no information at all
scores. F51 found the twins' floor is ~0.35, not 0, which changes how
their decoy gates read and showed one twin had been failing in the
direction a "collapses toward 0" reading scores as a pass.

Every other game in the battery has the same exposure. This measures the
floor for all of them, so no gate is ever read against an assumed zero
again.

Two floors per game, because "no information" has two forms a degenerate
policy can take:
  uniform    actions drawn uniformly at random every step
  fixed      the single best constant action, held all episode -- what a
             flat-logit argmax collapses to

The reportable floor is the max of the two: a gate must clear the best
thing an uninformed policy can do, not the average one.
"""

from __future__ import annotations

import argparse
import json

import torch

from experiments.games_amodal.fragment_bank import (
    battery_suite,
    compose_suite,
    mastery,
    twins_suite,
)
from experiments.games_amodal.game_family import FamilyConfig, FamilyVerifier
from experiments.games_amodal.shared_controller import SHARED_KEY_COUNT

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--steps", type=int, default=48)
parser.add_argument("--repeats", type=int, default=8)
args = parser.parse_args()


def score(config: FamilyConfig, policy, seed: int) -> float:
    verifier = FamilyVerifier(config, batch_size=args.batch_size, seed=seed)
    verifier.reset(seed=seed)
    rewards, masks = [], []
    alive = torch.ones(args.batch_size, dtype=torch.bool)
    for step in range(args.steps):
        masks.append(alive.float())
        outcome = verifier.step(policy(step))
        rewards.append(outcome.reward)
        alive = outcome.alive
    summary = {
        "total_reward": torch.stack(rewards, 1).sum(1),
        "mask": torch.stack(masks, 1),
    }
    # `mastery` scores dual games by per-rule accuracy, but ONLY when
    # rule_accuracy is present -- otherwise it silently falls through to
    # the reward branch, which credits an agent that engages every trial
    # and knows neither rule. Measured: that wrong branch reports random
    # play at 0.98 on dualAD, above its 0.686 solo ceiling. Supply the
    # verifier-side fields exactly as rollout_family does.
    if config.dual:
        summary["rule_accuracy"] = torch.tensor(verifier.dual_accuracy())
        summary["rule_engagement"] = (
            torch.tensor(verifier.dual_engagement())
            / max(args.batch_size, 1)
        )
    return float(mastery(summary, config))


def floors(config: FamilyConfig) -> dict[str, float]:
    uniform, fixed = [], []
    for repeat in range(args.repeats):
        seed = 7000 + repeat
        generator = torch.Generator().manual_seed(1000 + repeat)
        uniform.append(score(config, lambda _step: torch.randint(
            0, SHARED_KEY_COUNT, (args.batch_size,), generator=generator,
        ), seed))
        fixed.append(max(
            score(config, lambda _step, a=action: torch.full(
                (args.batch_size,), a, dtype=torch.long), seed)
            for action in range(SHARED_KEY_COUNT)
        ))
    uniform_mean = float(torch.tensor(uniform).mean())
    fixed_mean = float(torch.tensor(fixed).mean())
    return {
        "uniform": round(uniform_mean, 4),
        "best_fixed": round(fixed_mean, 4),
        "floor": round(max(uniform_mean, fixed_mean), 4),
    }


train, holdout = battery_suite()
twins, _ = twins_suite()
ctrain, choldout = compose_suite()
extras = [
    FamilyConfig(navigate=True, name="navigate1"),
    FamilyConfig(intercept=1, name="intercept1"),
    FamilyConfig(forage=1, name="forageA"),
    FamilyConfig(collect=1, name="collect1"),
]

seen: set[str] = set()
report: dict[str, dict[str, float]] = {}
for config in train + holdout + twins + ctrain + choldout + extras:
    if config.name in seen:
        continue
    seen.add(config.name)
    report[config.name] = floors(config)

print(json.dumps(report, indent=2, sort_keys=True))

"""What does a UNIFORMLY RANDOM policy score on each twin?

The decoy gate says a norm-matched noise fragment must collapse
performance "to chance". We have been reading choiceA's 0.90-0.97 under
decoy as a failure. But the ignorance diagnostic shows the decoy policy
is numerically almost exactly uniform (entropy 1.3859 vs ln(4)=1.38629,
max prob 0.2565 vs 0.25) while still scoring 0.969 greedy and 0.875
sampled. A near-uniform policy scoring 0.875 is only possible if chance
itself scores high on that twin.

So measure it: drive each twin with actions drawn uniformly at random,
no agent at all, and report mastery. That number is the true floor of
the decoy gate, and it may differ per twin.

If chance on choiceA is ~0.87, then "decoy collapses to chance" was
never achievable as we stated it, and F48/F49's reading of choiceA's
decoy as a failure has to be restated against the measured floor.
"""

from __future__ import annotations

import argparse
import json

import torch

from experiments.games_amodal.fragment_bank import mastery, twins_suite
from experiments.games_amodal.game_family import FamilyVerifier

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--steps", type=int, default=48)
parser.add_argument("--probe-steps", type=int, default=2)
parser.add_argument("--repeats", type=int, default=8)
parser.add_argument("--actions", type=int, default=4)
args = parser.parse_args()

train, _ = twins_suite()
report: dict[str, dict[str, float]] = {}

for config in train:
    uniform_scores, fixed_scores = [], []
    for repeat in range(args.repeats):
        # (a) uniformly random actions every step
        generator = torch.Generator().manual_seed(1000 + repeat)
        verifier = FamilyVerifier(config, batch_size=args.batch_size,
                                  seed=7000 + repeat)
        verifier.reset(seed=7000 + repeat)
        rewards, masks = [], []
        alive = torch.ones(args.batch_size, dtype=torch.bool)
        for _ in range(args.steps):
            masks.append(alive.float())
            actions = torch.randint(
                0, args.actions, (args.batch_size,), generator=generator
            )
            outcome = verifier.step(actions)
            rewards.append(outcome.reward)
            alive = outcome.alive
        uniform_scores.append(mastery(
            {"total_reward": torch.stack(rewards, 1).sum(1),
             "mask": torch.stack(masks, 1)}, config))

        # (b) a single fixed action for the whole episode, worst over the
        # action set -- the cheapest degenerate policy a flat-logit
        # argmax could land on.
        per_action = []
        for action in range(args.actions):
            verifier = FamilyVerifier(config, batch_size=args.batch_size,
                                      seed=7000 + repeat)
            verifier.reset(seed=7000 + repeat)
            rewards, masks = [], []
            alive = torch.ones(args.batch_size, dtype=torch.bool)
            for _ in range(args.steps):
                masks.append(alive.float())
                outcome = verifier.step(
                    torch.full((args.batch_size,), action, dtype=torch.long)
                )
                rewards.append(outcome.reward)
                alive = outcome.alive
            per_action.append(float(mastery(
                {"total_reward": torch.stack(rewards, 1).sum(1),
                 "mask": torch.stack(masks, 1)}, config)))
        fixed_scores.append(max(per_action))

    report[config.name] = {
        "uniform_random_mean": float(torch.tensor(uniform_scores).mean()),
        "uniform_random_max": float(torch.tensor(uniform_scores).max()),
        "best_fixed_action_mean": float(torch.tensor(fixed_scores).mean()),
    }

print(json.dumps(report, indent=2))

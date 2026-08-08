"""Does the co-trained loop's PROBE earn the score the bank is credited with?

The masked diagnostic ruled out my post-death explanation
(decoy_live_fraction = 1.0, so masked and unmasked entropy are equal).
The contradiction it leaves is sharp: under decoy the policy sits at
entropy 1.3852 against a maximum of 1.3863 and max-prob 0.2694 against
0.25 -- numerically almost uniform -- yet SAMPLING from it scores 0.9375
on choiceA, where uniformly random play scores 0.371 (F51).

A near-uniform policy cannot score 0.9375 if its actions are what earn
the reward. So something else is earning it.

Candidate: every episode opens with `probe_steps` steps of a FIXED,
hand-coded `test_action` that deliberately steps onto the positive-plane
item, so the agent can read the reward sign and infer which twin it is
in. But choiceA's rule IS "take the positive-plane item" -- so the probe
action performs choiceA's task. And mastery for a choice game is
`(total_reward > 0)`, which the probe alone can satisfy. On choiceB the
same probe action is exactly wrong, which would explain that twin
scoring BELOW chance.

If true, choiceA's decoy score measures the probe, not the bank, and the
twin asymmetry we have been attributing to a default policy in the
weights (F11/F48/F51) is an artifact of the harness.

Test: run the probe prefix, then act with NO policy at all -- uniformly
random, and separately a frozen no-op -- and score it the way the loop
does. Compared against the same thing with no probe prefix.
"""

from __future__ import annotations

import argparse
import json

import torch

from experiments.games_amodal.fragment_bank import mastery, twins_suite
from experiments.games_amodal.game_family import FamilyVerifier
from experiments.games_amodal.shared_controller import SHARED_KEY_COUNT

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--steps", type=int, default=48)
parser.add_argument("--probe-steps", type=int, default=2)
parser.add_argument("--repeats", type=int, default=8)
args = parser.parse_args()

DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def test_action(observation: torch.Tensor) -> torch.Tensor:
    """Verbatim from cotrained.py: step onto the positive-plane item."""
    batch = observation.shape[0]
    centre = observation.shape[-1] // 2
    actions = torch.zeros(batch, dtype=torch.long)
    for row in range(batch):
        for index, (dr, dc) in enumerate(DELTAS):
            if float(observation[row, 1, centre + dr, centre + dc]) > 0:
                actions[row] = index
                break
    return actions


def run(config, *, probe: bool, after: str, seed: int,
        post_only: bool = False) -> float:
    verifier = FamilyVerifier(config, batch_size=args.batch_size, seed=seed)
    verifier.reset(seed=seed)
    generator = torch.Generator().manual_seed(seed)
    rewards, masks = [], []
    alive = torch.ones(args.batch_size, dtype=torch.bool)
    for step in range(args.steps):
        masks.append(alive.float())
        if probe and step < args.probe_steps:
            actions = test_action(verifier.observation())
        elif after == "random":
            actions = torch.randint(
                0, SHARED_KEY_COUNT, (args.batch_size,), generator=generator
            )
        else:  # a single frozen action for the rest of the episode
            actions = torch.zeros(args.batch_size, dtype=torch.long)
        outcome = verifier.step(actions)
        rewards.append(outcome.reward)
        alive = outcome.alive
    reward_matrix = torch.stack(rewards, 1)
    mask_matrix = torch.stack(masks, 1)
    if post_only:
        # Candidate fix: score only what happened AFTER the probe, so the
        # hand-coded probe action cannot pay for the episode.
        reward_matrix = reward_matrix[:, args.probe_steps:]
        mask_matrix = mask_matrix[:, args.probe_steps:]
    return float(mastery(
        {"total_reward": reward_matrix.sum(1), "mask": mask_matrix}, config))


train, _ = twins_suite()
report: dict[str, dict[str, float]] = {}
for config in train:
    row: dict[str, float] = {}
    for post_only in (False, True):
        for probe in (False, True):
            for after in ("random", "frozen"):
                scores = [
                    run(config, probe=probe, after=after, seed=7000 + repeat,
                        post_only=post_only)
                    for repeat in range(args.repeats)
                ]
                key = (f"{'POSTONLY ' if post_only else ''}"
                       f"{'probe+' if probe else 'no-probe+'}{after}")
                row[key] = round(float(torch.tensor(scores).mean()), 4)
    report[config.name] = row

print(json.dumps(report, indent=2))

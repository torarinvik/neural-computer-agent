"""HORIZON CEILING: how much is left on this family at ANY depth?

F229's instrument: the deployable depth-1 proxy stack BEATS privileged
2-step true-return planning on the event worlds (collect2 +2.57 vs
+0.76; intercept2 +0.15 vs -0.64) -- dense learned shaping encodes
structure beyond any shallow lookahead. So the true task ceiling was
never measured. This probe runs exhaustive true-return search under
the real simulator at depths 1-4 (privileged diagnostic, 4^d rollout
copies per step). If depth-4 truth only modestly exceeds the proxy
stack, the family is SATURATED and the bottleneck moves to world
diversity (the sealed mechanism benchmark). If depth-4 truth is far
above, horizon machinery (n-step value carriers, not 2-step argmax)
returns to the frontier with a measured prize.
"""

from __future__ import annotations

import argparse
import copy
import json

import torch

from experiments.games_amodal.game_family import FamilyConfig, FamilyVerifier

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=69316)
parser.add_argument("--episodes", type=int, default=64)
parser.add_argument("--steps", type=int, default=12)
parser.add_argument("--max-depth", type=int, default=4)
parser.add_argument("--json", default="")
args = parser.parse_args()

torch.set_num_threads(1)


def lookahead(v, depth, episodes):
    best = None
    for act in range(4):
        shadow = copy.deepcopy(v)
        gain = shadow.step(torch.full((episodes,), act)).reward
        if depth > 1:
            gain = gain + lookahead(shadow, depth - 1, episodes)
        best = gain if best is None else torch.maximum(best, gain)
    return best


def play_true(config, seed, episodes, steps, depth):
    v = FamilyVerifier(config, batch_size=episodes, seed=seed)
    v.reset(seed=seed)
    total = torch.zeros(episodes)
    for _ in range(steps):
        best, action = None, torch.zeros(episodes, dtype=torch.long)
        for act in range(4):
            shadow = copy.deepcopy(v)
            gain = shadow.step(torch.full((episodes,), act)).reward
            if depth > 1:
                gain = gain + lookahead(shadow, depth - 1, episodes)
            if best is None:
                best = gain.clone()
            else:
                take = gain > best
                best = torch.where(take, gain, best)
                action = torch.where(take, torch.full((episodes,), act),
                                     action)
        total += v.step(action).reward
    return float(total.mean())


WORLDS = [
    ("pursue1", FamilyConfig(pursue=1)),
    ("pursue1_avoid1", FamilyConfig(pursue=1, avoid=1)),
    ("pursue1_avoid2", FamilyConfig(pursue=1, avoid=2)),
    ("pursue1_collect1", FamilyConfig(pursue=1, collect=1)),
    ("avoid2", FamilyConfig(avoid=2)),
    ("avoid3", FamilyConfig(avoid=3)),
    ("intercept2", FamilyConfig(intercept=2)),
    ("collect2", FamilyConfig(collect=2)),
]

report = {"seed": args.seed, "results": {}}
for name, config in WORLDS:
    row = {}
    for depth in range(1, args.max_depth + 1):
        row[f"true_d{depth}"] = play_true(config, args.seed * 977,
                                          args.episodes, args.steps, depth)
    report["results"][name] = row
    print(f"  {name:<16} " + "  ".join(
        f"d{d} {row[f'true_d{d}']:+.3f}" for d in range(1, args.max_depth + 1)),
        flush=True)

print(json.dumps(report, indent=2))
if args.json:
    with open(args.json, "w") as handle:
        json.dump(report, handle, indent=2)

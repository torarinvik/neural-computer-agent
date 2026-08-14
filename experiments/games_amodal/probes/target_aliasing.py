"""TARGET-ALIASING AUDIT (F233 prelude, GPT-advised).

Question: can any representation AVAILABLE to the current stack
distinguish plane-1 items with different contact values on
deceptive1, and which feature level first separates them?

Levels, progressively privileged:
  L1 rank            was the contacted item the nearest plane-1 item?
  L2 hazard-now      Manhattan distance item -> hazard at contact time
  L3 hazard-row      |item_row - hazard_row| (static relational: the
                     hazard patrols one row, so this is stable)
  L4 spawn history   distance item -> hazard AT THE ITEM'S SPAWN
                     (requires persistent identity; privileged here)

For each level: pairwise ranking accuracy separating bait (+0.2)
contacts from food (+1.0) contacts. The first level that separates
names the missing capability: L2/L3 -> a relational feature suffices
(no entity tracking); only-L4 -> persistent identity is required.
"""

from __future__ import annotations

import argparse
import json

import torch

from experiments.games_amodal.game_family import FamilyConfig, FamilyVerifier

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=69316)
parser.add_argument("--episodes", type=int, default=256)
parser.add_argument("--steps", type=int, default=12)
parser.add_argument("--rounds", type=int, default=8)
parser.add_argument("--json", default="")
args = parser.parse_args()

torch.set_num_threads(1)
config = FamilyConfig(avoid=1, collect=1, deceptive=1)

events = []  # (is_food, rank_nearest, d_hazard_now, d_hazard_row, d_spawn)
g = torch.Generator().manual_seed(args.seed)
for r in range(args.rounds):
    v = FamilyVerifier(config, batch_size=args.episodes,
                       seed=args.seed + r * 101)
    v.reset(seed=args.seed + r * 101)
    spawn_dist = {}
    for row in range(args.episodes):
        hz = v._hazards[row][0]
        for cell in v._food[row]:
            spawn_dist[(row, cell)] = abs(cell[0] - hz[0]) + abs(
                cell[1] - hz[1])
        for cell in v._bait[row]:
            spawn_dist[(row, cell)] = abs(cell[0] - hz[0]) + abs(
                cell[1] - hz[1])
    for _ in range(args.steps):
        before_food = [list(v._food[row]) for row in range(args.episodes)]
        before_bait = [list(v._bait[row]) for row in range(args.episodes)]
        before_haz = [v._hazards[row][0] for row in range(args.episodes)]
        before_av = list(v._avatar)
        actions = torch.randint(0, 4, (args.episodes,), generator=g)
        out = v.step(actions)
        for row in range(args.episodes):
            target = v._avatar[row]
            pool = before_food[row] + before_bait[row]
            if target not in pool:
                continue
            is_food = target in before_food[row]
            av = before_av[row]
            dists = [abs(c[0] - av[0]) + abs(c[1] - av[1]) for c in pool]
            mine = abs(target[0] - av[0]) + abs(target[1] - av[1])
            rank_nearest = 1.0 if mine == min(dists) else 0.0
            hz = before_haz[row]
            d_now = abs(target[0] - hz[0]) + abs(target[1] - hz[1])
            d_row = abs(target[0] - hz[0])
            d_spawn = spawn_dist.get((row, target), d_now)
            # track respawns for spawn-distance bookkeeping
            hz2 = v._hazards[row][0]
            for cell in v._food[row] + v._bait[row]:
                if (row, cell) not in spawn_dist:
                    spawn_dist[(row, cell)] = abs(cell[0] - hz2[0]) + abs(
                        cell[1] - hz2[1])
            events.append((is_food, rank_nearest, float(d_now),
                           float(d_row), float(d_spawn)))

food = [e for e in events if e[0]]
bait = [e for e in events if not e[0]]
print(f"{len(events)} contacts: {len(food)} food, {len(bait)} bait")


def ranking_accuracy(idx, larger_is_food=True):
    """P(food_feature > bait_feature) over all pairs, ties = 0.5."""
    wins = ties = 0
    for f in food:
        for b in bait:
            if f[idx] == b[idx]:
                ties += 1
            elif (f[idx] > b[idx]) == larger_is_food:
                wins += 1
    total = len(food) * len(bait)
    return (wins + 0.5 * ties) / total if total else float("nan")


report = {"seed": args.seed, "contacts": len(events),
          "food": len(food), "bait": len(bait),
          "L1_rank": round(ranking_accuracy(1), 4),
          "L2_hazard_now": round(ranking_accuracy(2), 4),
          "L3_hazard_row": round(ranking_accuracy(3), 4),
          "L4_spawn": round(ranking_accuracy(4), 4)}
for k in ("L1_rank", "L2_hazard_now", "L3_hazard_row", "L4_spawn"):
    print(f"  {k:<14} ranking accuracy {report[k]}")
print(json.dumps(report))
if args.json:
    with open(args.json, "w") as handle:
        json.dump(report, handle, indent=2)

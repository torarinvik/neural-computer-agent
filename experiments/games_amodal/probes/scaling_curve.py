"""SAMPLE-COMPLEXITY SCALING CURVE: is experience scale a live
route through the fenced residual? (F261)

F260 closed the fence: five mechanism-level attacks on the trio
residual refuted; the priced routes left are (a) orders-more
experience and (b) a stochastic learned simulator. This probe
prices (a) with F255's own instrument: the knn_full distillation
policy (privileged features, so state is not the confound) at data
multipliers 1x / 3x / 9x (4, 12, 36 rollouts of 256 episodes x 12
steps of true depth-2 Q targets).

Registered predictions (before any run; an either/or):
  P1 if knn_full improves by >= +0.15 from 1x to 9x on >= 2/3 trio
     worlds, SCALE IS A LIVE ROUTE (the curve, extrapolated, prices
     the experience needed to reach the anchors); if the gain is
     <= +0.05, scale is DEAD at feasible budgets and the stochastic
     simulator becomes the only deployable route.
  P2 control world: flat (already at the anchor at 1x).
  P3 the privileged d2 anchor is scale-independent (sanity).
"""

from __future__ import annotations

import argparse
import copy
import json

import torch

from experiments.games_amodal.game_family import FamilyConfig, FamilyVerifier

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=69316)
parser.add_argument("--data-episodes", type=int, default=192)
parser.add_argument("--data-rollouts", type=int, default=3)
parser.add_argument("--epsilon", type=float, default=0.3)
parser.add_argument("--eval-episodes", type=int, default=64)
parser.add_argument("--steps", type=int, default=12)
parser.add_argument("--knn", type=int, default=8)
parser.add_argument("--match-dist", type=int, default=2)
parser.add_argument("--json", default="")
args = parser.parse_args()

torch.set_num_threads(1)

SLOTS, VALUES = 8, 8
ABSENT = VALUES
HEIGHT = WIDTH = 8
PLANES = 3
TRACKED = ((1, 2), (2, 4), (2, 6))
CAPS = {"food": 3, "fallers": 2, "hazards": 3, "pursuers": 2,
        "resources": 2}
FULL_DIM = 2 + 2 * sum(CAPS.values()) + 2 + 1 + 1 + 1


def frame_cells(obs):
    return obs.view(-1, PLANES, HEIGHT, WIDTH)


class Tracker:
    def __init__(self, batch):
        self.pos = [[None, None, None] for _ in range(batch)]

    def encode(self, frames):
        B = frames.shape[0]
        out = torch.full((B, SLOTS), ABSENT, dtype=torch.long)
        avatar = frames[:, 0].reshape(B, -1)
        present = avatar.max(dim=1).values > 0
        flat = avatar.argmax(dim=1)
        for b in range(B):
            if not bool(present[b]):
                self.pos[b] = [None, None, None]
                continue
            ar, ac = int(flat[b]) // WIDTH, int(flat[b]) % WIDTH
            out[b, 0], out[b, 1] = ar, ac
            claimed = set()
            for t_ix, (plane, base) in enumerate(TRACKED):
                cells = [(int(r), int(c)) for r, c in
                         (frames[b, plane] > 0).nonzero()
                         if (plane, int(r), int(c)) not in claimed]
                prev = self.pos[b][t_ix]
                pick = None
                if prev is not None and cells:
                    near = min(cells, key=lambda z: abs(z[0] - prev[0])
                               + abs(z[1] - prev[1]))
                    if (abs(near[0] - prev[0]) + abs(near[1] - prev[1])
                            <= args.match_dist):
                        pick = near
                if pick is None and cells:
                    rank = 0 if base != 6 else 1
                    cells_sorted = sorted(
                        cells, key=lambda z: abs(z[0] - ar)
                        + abs(z[1] - ac))
                    if len(cells_sorted) > rank:
                        pick = cells_sorted[rank]
                self.pos[b][t_ix] = pick
                if pick is not None:
                    claimed.add((plane, pick[0], pick[1]))
                    out[b, base], out[b, base + 1] = pick
        return out


def q_true_d2(v, episodes):
    Q = torch.zeros(episodes, 4)
    for act in range(4):
        s1 = copy.deepcopy(v)
        r1 = s1.step(torch.full((episodes,), act)).reward
        best2 = None
        for act2 in range(4):
            s2 = copy.deepcopy(s1)
            r2 = s2.step(torch.full((episodes,), act2)).reward
            best2 = r2 if best2 is None else torch.maximum(best2, r2)
        Q[:, act] = r1 + best2
    return Q


def full_features(v, episodes):
    """[E, FULL_DIM] privileged features: all entity positions
    (padded with -1), switch, holding, pending count, alive."""
    out = torch.full((episodes, FULL_DIM), -1.0)
    for e in range(episodes):
        i = 0
        out[e, i:i + 2] = torch.tensor(v._avatar[e], dtype=torch.float)
        i += 2
        for attr, cap in (("_food", CAPS["food"]),
                          ("_fallers", CAPS["fallers"]),
                          ("_hazards", CAPS["hazards"]),
                          ("_pursuers", CAPS["pursuers"]),
                          ("_resources", CAPS["resources"])):
            items = getattr(v, attr)[e]
            for k in range(cap):
                if k < len(items):
                    out[e, i] = float(items[k][0])
                    out[e, i + 1] = float(items[k][1])
                i += 2
        sw = v._switches[e]
        if sw is not None:
            out[e, i] = float(sw[0])
            out[e, i + 1] = float(sw[1])
        i += 2
        holding = getattr(v, "_holding", None)
        out[e, i] = float(bool(holding[e])) if holding is not None \
            else -1.0
        i += 1
        pending = getattr(v, "_pending", None)
        out[e, i] = float(len(pending[e])) if pending is not None \
            else -1.0
        i += 1
        out[e, i] = float(bool(v._alive[e]))
    return out


def knn_q(query, data_x, data_q):
    """Inverse-distance k-NN regression of Q, [Nq, 4]."""
    d = (query.unsqueeze(1) - data_x.unsqueeze(0)).abs().sum(dim=2)
    k = min(args.knn, data_x.shape[0])
    near_d, near_i = d.topk(k, dim=1, largest=False)
    w = 1.0 / (near_d + 1.0)
    q = (data_q[near_i] * w.unsqueeze(2)).sum(dim=1) \
        / w.sum(dim=1, keepdim=True)
    return q


def collect(config, seed):
    """Rollouts under the privileged d2 policy with epsilon
    exploration; returns tracked states, full features, Q targets."""
    E = args.data_episodes
    g = torch.Generator().manual_seed(seed + 8888)
    xs_t, xs_f, qs = [], [], []
    for r in range(args.data_rollouts):
        v = FamilyVerifier(config, batch_size=E, seed=seed + r)
        v.reset(seed=seed + r)
        tracker = Tracker(E)
        for _ in range(args.steps):
            code = tracker.encode(frame_cells(v.observation()))
            Q = q_true_d2(v, E)
            xs_t.append(code.float())
            xs_f.append(full_features(v, E))
            qs.append(Q)
            action = Q.argmax(dim=1)
            explore = torch.rand(E, generator=g) < args.epsilon
            action = torch.where(
                explore, torch.randint(0, 4, (E,), generator=g),
                action)
            v.step(action)
    return torch.cat(xs_t), torch.cat(xs_f), torch.cat(qs)


def play(config, mode, seed, data=None):
    E = args.eval_episodes
    v = FamilyVerifier(config, batch_size=E, seed=seed)
    v.reset(seed=seed)
    g = torch.Generator().manual_seed(seed + 4242)
    tracker = Tracker(E)
    total = torch.zeros(E)
    for _ in range(args.steps):
        code = tracker.encode(frame_cells(v.observation()))
        if mode == "random":
            action = torch.randint(0, 4, (E,), generator=g)
        elif mode == "privileged":
            action = q_true_d2(v, E).argmax(dim=1)
        elif mode == "knn_tracked":
            action = knn_q(code.float(), data[0], data[2]).argmax(dim=1)
        elif mode == "knn_full":
            action = knn_q(full_features(v, E), data[1],
                           data[2]).argmax(dim=1)
        total += v.step(action).reward
    return float(total.mean())


WORLDS = [
    ("collect1_intercept1_pursue1_resource1",
     FamilyConfig(collect=1, intercept=1, pursue=1, resource=1)),
    ("delayed3_intercept1_pursue1_resource2",
     FamilyConfig(delayed=3, intercept=1, pursue=1, resource=2)),
    ("delayed3_intercept2_pursue1_resource1",
     FamilyConfig(delayed=3, intercept=2, pursue=1, resource=1)),
    ("ctrl_avoid1_collect1", FamilyConfig(avoid=1, collect=1)),
]

report = {"seed": args.seed, "results": {}}
for name, config in WORLDS:
    row = {"random": play(config, "random", args.seed * 977),
           "privileged_d2": play(config, "privileged",
                                 args.seed * 977)}
    for mult, rollouts in (("1x", 4), ("3x", 12), ("9x", 36)):
        args.data_rollouts = rollouts
        data = collect(config, args.seed * 31)
        row[f"knn_full_{mult}"] = play(config, "knn_full",
                                       args.seed * 977, data)
        row[f"n_{mult}"] = int(data[0].shape[0])
    report["results"][name] = row
    print(f"  {name:<40} priv {row['privileged_d2']:+.3f}  1x "
          f"{row['knn_full_1x']:+.3f}  3x {row['knn_full_3x']:+.3f}"
          f"  9x {row['knn_full_9x']:+.3f}", flush=True)

print(json.dumps(report, indent=2))
if args.json:
    with open(args.json, "w") as handle:
        json.dump(report, handle, indent=2)

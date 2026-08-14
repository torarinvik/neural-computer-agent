"""TRACKED-STATE CAPACITY AUDIT, v2: the distillation instrument
(F255).

Five integrations (F250-F254) improved every learned layer in
isolation and all stalled at deploy parity ~-0.85 on the trio while
privileged arms certify reachable skill at -0.16..-0.46. The last
untested hypothesis is the one the freeze discipline reserves for
strongest evidence: the CORE's 8-slot state (avatar + three tracked
entities, no holding/pending bits) cannot carry the decision -- the
trio worlds hold 4-5 non-avatar entities plus hidden mechanism
state.

v1 (exact-key bucketing) was stillborn: 8-slot keys never collide
at feasible batch sizes (coverage ~0). The v2 instrument holds the
MACHINERY fixed and varies only the STATE: one nonparametric
regressor (k-NN over true depth-2 action values, L1 distance,
inverse-distance vote), fit on data gathered under the privileged
policy plus exploration, twice --

  knn_tracked   inputs = the tracked 8-slot state (the core's view)
  knn_full      inputs = privileged full features (every entity
                position, holding, pending, padded fixed-size)

-- and both evaluated AS POLICIES on a fresh stream. The paired gap
knn_full - knn_tracked is the capacity charge at matched machinery;
the privileged d2 policy value anchors the instrument.

Registered predictions (before any run; an either/or):
  P1 if knn_full - knn_tracked >= 0.25 on >= 2/3 trio worlds,
     CAPACITY IS CERTIFIED as the binding constraint -- the
     necessity witness for core state expansion. If the gap is
     <= 0.10 on >= 2/3, the core is EXONERATED and machinery (data,
     heads, search) stays on the hook.
  P2 control world: gap ~ 0 (the learned stack already exceeds the
     privileged ceiling there -- the state must suffice).
  P3 instrument validity: knn_full within 0.20 of the privileged
     d2 anchor on >= 3/4 worlds; otherwise the regressor class is
     too weak and the audit is INCONCLUSIVE (logged, not spun).
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
    data = collect(config, args.seed * 31)
    row = {"random": play(config, "random", args.seed * 977),
           "privileged_d2": play(config, "privileged", args.seed * 977),
           "knn_tracked": play(config, "knn_tracked", args.seed * 977,
                               data),
           "knn_full": play(config, "knn_full", args.seed * 977,
                            data),
           "n_data": int(data[0].shape[0])}
    row["capacity_gap"] = round(row["knn_full"] - row["knn_tracked"], 4)
    report["results"][name] = row
    print(f"  {name:<40} rnd {row['random']:+.3f}  priv "
          f"{row['privileged_d2']:+.3f}  knn_full "
          f"{row['knn_full']:+.3f}  knn_tracked "
          f"{row['knn_tracked']:+.3f}  GAP {row['capacity_gap']:+.3f}",
          flush=True)

print(json.dumps(report, indent=2))
if args.json:
    with open(args.json, "w") as handle:
        json.dump(report, handle, indent=2)

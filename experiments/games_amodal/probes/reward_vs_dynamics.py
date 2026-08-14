"""REWARD HEAD vs DYNAMICS: the one-cell split of F250's residual
(F251).

F250's tracked VALUE-PLAN sits at deploy parity on the trio while
F247's privileged arm (true dynamics + TRUE immediate reward + the
same learned value head) reached -0.34..-0.46. The ~+0.45 between
them belongs to {learned dynamics, learned reward head}. This
privileged diagnostic separates the two WITHOUT building anything:

  truedyn_truer_d2    true dynamics, true edge rewards, learned V
                      at the leaves (F247's arm, re-anchored here)
  truedyn_learnr_d2   true dynamics, but edge rewards from the
                      LEARNED head r_hat(s, a) over tracked slots;
                      learned V at the leaves

Both plan depth 2 over deep-copied simulators with a cloned tracker
for shadow perception. The only difference is the edge score.

Registered predictions (before any run; a genuine either/or):
  P1 anchor: truedyn_truer_d2 lands within +-0.10 of F247's
     truedyn_d2 per world (protocol sanity).
  P2 the split: if truedyn_learnr_d2 >= truedyn_truer_d2 - 0.15 the
     reward head is exonerated and DYNAMICS FIDELITY is indicted;
     if truedyn_learnr_d2 <= random + 0.10 the REWARD BINDING is
     indicted (type-aliased tracked groups; the F235 machinery is
     the response). In between: both contribute, sized by the gap.
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
parser.add_argument("--data-batch", type=int, default=192)
parser.add_argument("--data-steps", type=int, default=20)
parser.add_argument("--horizon", type=int, default=8)
parser.add_argument("--ridge-lam", type=float, default=1.0)
parser.add_argument("--match-dist", type=int, default=2)
parser.add_argument("--json", default="")
args = parser.parse_args()

torch.set_num_threads(1)

SLOTS, VALUES = 8, 8
ABSENT = VALUES
HEIGHT = WIDTH = 8
PLANES = 3
GROUPS = ((0, 1), (2, 3), (4, 5), (6, 7))
GROUP_PAIRS = [(i, j) for i in range(4) for j in range(i + 1, 4)]
FEATS = 1 + SLOTS + 2 * len(GROUP_PAIRS)


def frame_cells(obs):
    return obs.view(-1, PLANES, HEIGHT, WIDTH)


class Tracker:
    def __init__(self, batch):
        self.pos = [[None, None, None] for _ in range(batch)]

    def clone(self):
        out = Tracker(len(self.pos))
        out.pos = [list(p) for p in self.pos]
        return out

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
            for t_ix, (plane, base) in enumerate(
                    ((1, 2), (2, 4), (2, 6))):
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


def clamp_state(code):
    return torch.where(code < VALUES, code, torch.zeros_like(code))


def phi(states):
    s = states.float()
    feats = [torch.ones(s.shape[0], 1), s]
    for i, j in GROUP_PAIRS:
        gi, gj = GROUPS[i], GROUPS[j]
        d = ((s[:, gi[0]] - s[:, gj[0]]).abs()
             + (s[:, gi[1]] - s[:, gj[1]]).abs())
        feats.append(d.unsqueeze(1))
        feats.append((d <= 1).float().unsqueeze(1))
    return torch.cat(feats, dim=1)


def ridge(X, y, lam):
    A = X.T @ X + lam * torch.eye(X.shape[1])
    return torch.linalg.solve(A, X.T @ y)


def collect_experience(config, seed):
    B, T, H = args.data_batch, args.data_steps, args.horizon
    v = FamilyVerifier(config, batch_size=B, seed=seed)
    v.reset(seed=seed)
    g = torch.Generator().manual_seed(seed + 31337)
    tracker = Tracker(B)
    states, actions, rewards = [], [], []
    code = clamp_state(tracker.encode(frame_cells(v.observation())))
    for _ in range(T):
        action = torch.randint(0, 4, (B,), generator=g)
        states.append(code)
        actions.append(action)
        rewards.append(v.step(action).reward)
        code = clamp_state(tracker.encode(frame_cells(v.observation())))
    S = torch.stack(states)
    A = torch.stack(actions)
    R = torch.stack(rewards)
    ret = torch.zeros(T, B)
    for t in range(T):
        ret[t] = R[t:min(T, t + H)].sum(dim=0)
    keep = T - H if T > H else T
    return (S[:keep].reshape(-1, SLOTS), A[:keep].reshape(-1),
            R[:keep].reshape(-1), ret[:keep].reshape(-1))


def fit_heads(S, A, R, ret):
    X = phi(S)
    w_r = torch.zeros(4, FEATS)
    for act in range(4):
        rows = A == act
        if int(rows.sum()) >= FEATS:
            w_r[act] = ridge(X[rows], R[rows], args.ridge_lam)
    w_v = ridge(X, ret, args.ridge_lam)
    return w_r, w_v


def play(config, mode, w_r, w_v, seed):
    """mode: random | truer | learnr (both true-dynamics depth 2)."""
    episodes, steps = args.episodes, args.steps
    v = FamilyVerifier(config, batch_size=episodes, seed=seed)
    v.reset(seed=seed)
    g = torch.Generator().manual_seed(seed + 4242)
    tracker = Tracker(episodes)
    total = torch.zeros(episodes)
    for _ in range(steps):
        frames = frame_cells(v.observation())
        code = clamp_state(tracker.encode(frames))
        if mode == "random":
            action = torch.randint(0, 4, (episodes,), generator=g)
            total += v.step(action).reward
            continue
        X0 = phi(code)
        best, action = None, torch.zeros(episodes, dtype=torch.long)
        for act in range(4):
            s1 = copy.deepcopy(v)
            r1_true = s1.step(torch.full((episodes,), act)).reward
            t1 = tracker.clone()
            code1 = clamp_state(t1.encode(frame_cells(s1.observation())))
            r1 = r1_true if mode == "truer" else X0 @ w_r[act]
            X1 = phi(code1)
            sub = None
            for act2 in range(4):
                s2 = copy.deepcopy(s1)
                r2_true = s2.step(torch.full((episodes,), act2)).reward
                t2 = t1.clone()
                code2 = clamp_state(
                    t2.encode(frame_cells(s2.observation())))
                r2 = r2_true if mode == "truer" else X1 @ w_r[act2]
                leaf = r1 + r2 + phi(code2) @ w_v
                sub = leaf if sub is None else torch.maximum(sub, leaf)
            if best is None:
                best = sub.clone()
            else:
                take = sub > best
                best = torch.where(take, sub, best)
                action = torch.where(
                    take, torch.full((episodes,), act), action)
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
    S, A, R, ret = collect_experience(config, args.seed * 53)
    w_r, w_v = fit_heads(S, A, R, ret)
    row = {"random": play(config, "random", None, None, args.seed * 977),
           "truedyn_truer_d2": play(config, "truer", w_r, w_v,
                                    args.seed * 977),
           "truedyn_learnr_d2": play(config, "learnr", w_r, w_v,
                                     args.seed * 977)}
    report["results"][name] = row
    print(f"  {name:<40} " + "  ".join(
        f"{k} {v:+.3f}" for k, v in row.items()), flush=True)

print(json.dumps(report, indent=2))
if args.json:
    with open(args.json, "w") as handle:
        json.dump(report, handle, indent=2)

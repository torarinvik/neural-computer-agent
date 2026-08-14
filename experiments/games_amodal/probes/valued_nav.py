"""VALUE-DIRECTED NAVIGATION: transferred values x the substrate's
own gradient = complete zero-shot control (F271).

F270 exposed the seam: the class binding knows WHAT is valuable but
navigates myopically (kernel range 2), so hand-written approach-food
baselines outrun it at distance while ignoring hazards. The
composition: the TRANSFERRED values pick targets, the substrate's
native metric supplies the gradient --

    score(action) = sum_g v_hat(psi_g) * (d_now(g) - d_after(g))

(approach what the binding prices positive, retreat from what it
prices negative, in whatever metric the substrate has). Zero target
data anywhere: values from the grid, gradients from the substrate's
own distance.

Arms on ring worlds (rcollect1, ravoid1_collect1): random | gg
(hand-written approach-food) | class_kernel (F270 anchor) |
valued_nav | valued_nav_shuffled.

Registered predictions (before any run):
  P1 valued_nav matches the hand-written approach form on the pure
     collect world (>= gg - 0.15), 6/6 -- same gradient, chosen by
     transferred values instead of wiring.
  P2 THE PRIZE: on ravoid1_collect1 valued_nav BEATS gg by >= +0.3
     on >= 5/6 seeds (gg ignores the hazard; the transferred
     negative value avoids it). Negative terms are proximity-
     weighted (threat is contact-local -- F233's guard insight,
     value-directed); set during the design smoke, before the
     measurement run.
  P3 shuffled values collapse or invert valued_nav on >= 4/6.
"""

from __future__ import annotations

import argparse
import json

import torch

from experiments.games_amodal.game_family import FamilyConfig, FamilyVerifier
from experiments.games_amodal.graph_world import (
    GraphConfig, GraphVerifier, NODES, PORTS)

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
torch.manual_seed(args.seed)

SLOTS, VALUES = 8, 8
ABSENT = VALUES
HEIGHT = WIDTH = 8
PLANES = 3

def frame_cells(obs):
    return obs.view(-1, PLANES, HEIGHT, WIDTH)


TRACKED = ((1, 2), (2, 4), (2, 6))       # (plane, base slot) per entity
PSI_DIM = 5                              # 1, plane1, plane2, motion, approach


class Tracker:
    def __init__(self, batch):
        self.pos = [[None, None, None] for _ in range(batch)]
        self.steps = torch.zeros(batch, 3)
        self.move = torch.zeros(batch, 3)
        self.closer = torch.zeros(batch, 3)

    def clone(self):
        out = Tracker(len(self.pos))
        out.pos = [list(p) for p in self.pos]
        out.steps = self.steps.clone()
        out.move = self.move.clone()
        out.closer = self.closer.clone()
        return out

    def psi(self):
        """[B, 3, PSI_DIM] generic type signatures."""
        B = len(self.pos)
        steps = self.steps.clamp(min=1)
        out = torch.zeros(B, 3, PSI_DIM)
        out[:, :, 0] = 1.0
        for t_ix, (plane, _base) in enumerate(TRACKED):
            out[:, t_ix, 1] = 1.0 if plane == 1 else 0.0
            out[:, t_ix, 2] = 1.0 if plane == 2 else 0.0
        out[:, :, 3] = self.move / steps
        out[:, :, 4] = self.closer / steps
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
                if prev is not None and pick is not None:
                    d_move = (abs(pick[0] - prev[0])
                              + abs(pick[1] - prev[1]))
                    if d_move <= args.match_dist:   # identity held
                        self.steps[b, t_ix] += 1
                        self.move[b, t_ix] += d_move
                        d_prev = abs(prev[0] - ar) + abs(prev[1] - ac)
                        d_now = abs(pick[0] - ar) + abs(pick[1] - ac)
                        if d_now < d_prev:
                            self.closer[b, t_ix] += 1
                self.pos[b][t_ix] = pick
                if pick is not None:
                    claimed.add((plane, pick[0], pick[1]))
                    out[b, base], out[b, base + 1] = pick
        return out


def clamp_state(code):
    return torch.where(code < VALUES, code, torch.zeros_like(code))


# ---- heads ---------------------------------------------------------

GROUPS = ((0, 1), (2, 3), (4, 5), (6, 7))
GROUP_PAIRS = [(i, j) for i in range(4) for j in range(i + 1, 4)]
V_FEATS = 1 + SLOTS + 2 * len(GROUP_PAIRS)
EVENTS = 4                               # contact, teleport, both, boundary
R_FEATS = 3 * EVENTS * PSI_DIM + 4 + 1   # groups x events x psi + act + bias


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


def event_features(parent, child, actions, psi, died=None,
                   soft=False):
    """[N, R_FEATS] typed event features for parent->child transitions.

    psi: [N, 3, PSI_DIM].  died: bool [N] rows where the avatar
    vanished at child time (death attribution: contact := entity
    within 2 at parent).  soft=True replaces the binary contact
    predicate with the graded kernel clamp((2 - d)/2, 0, 1), so a
    one-cell dynamics error degrades the score instead of flipping
    it (F253)."""
    N = parent.shape[0]
    p, c = parent.float(), child.float()
    av_r, av_c = c[:, 0], c[:, 1]
    feats = [torch.ones(N, 1)]
    for t_ix, (_plane, base) in enumerate(TRACKED):
        er, ec = c[:, base], c[:, base + 1]
        pr, pc = p[:, base], p[:, base + 1]
        d_now = (av_r - er).abs() + (av_c - ec).abs()
        d_prev_cell = (av_r - pr).abs() + (av_c - pc).abs()
        if soft:
            contact = torch.maximum(
                ((2.0 - d_now) / 2.0).clamp(0.0, 1.0),
                (d_prev_cell == 0).float())
        else:
            contact = ((d_now <= 1) | (d_prev_cell == 0)).float()
        if died is not None:
            d_at_parent = ((p[:, 0] - pr).abs()
                           + (p[:, 1] - pc).abs())
            contact = torch.where(died, (d_at_parent <= 2).float(),
                                  contact)
        jump = (er - pr).abs() + (ec - pc).abs()
        teleport = (jump > 2).float()
        if died is not None:
            teleport = teleport * (~died).float()
        both = contact * teleport
        contact_only = contact * (1.0 - teleport)
        boundary = ((er == 0) | (er == HEIGHT - 1)
                    | (ec == 0) | (ec == WIDTH - 1)).float()
        ev = torch.stack([contact_only, teleport, both, boundary],
                         dim=1)
        feats.append((ev.unsqueeze(2) * psi[:, t_ix].unsqueeze(1))
                     .reshape(N, EVENTS * PSI_DIM))
    feats.append(torch.nn.functional.one_hot(actions, 4).float())
    return torch.cat(feats, dim=1)


def ridge(X, y, lam):
    A = X.T @ X + lam * torch.eye(X.shape[1])
    return torch.linalg.solve(A, X.T @ y)


# ---- data ----------------------------------------------------------

def collect_experience(config, seed, policy=None):
    """Tracked rollout recording states, psi snapshots, actions,
    rewards, avatar-vanish flags, and H-step returns."""
    B, T, H = args.data_batch, args.data_steps, args.horizon
    v = FamilyVerifier(config, batch_size=B, seed=seed)
    v.reset(seed=seed)
    g = torch.Generator().manual_seed(seed + 31337)
    tracker = Tracker(B)
    raw, psis, actions, rewards, alive = [], [], [], [], []
    code = tracker.encode(frame_cells(v.observation()))
    raw.append(code)
    psis.append(tracker.psi())
    alive.append(code[:, 0] < VALUES)
    for _ in range(T):
        state = clamp_state(code)
        if policy is None:
            action = torch.randint(0, 4, (B,), generator=g)
        else:
            action = policy(state, tracker.psi())
        actions.append(action)
        rewards.append(v.step(action).reward)
        code = tracker.encode(frame_cells(v.observation()))
        raw.append(code)
        psis.append(tracker.psi())
        alive.append(code[:, 0] < VALUES)
    return raw, psis, actions, rewards, alive


def build_head_data(raw, psis, actions, rewards, alive, soft=False,
                    bank=None):
    """Event-feature rows from a recorded rollout.  A row qualifies
    if the avatar was present at t; death at t+1 flows into the died
    flag (attribution).  With bank set, the child state is the
    BANK-ROLLED prediction from the parent for the taken action --
    model-consistent training (F253): the head is fit in the exact
    representation the planner will hand it."""
    xs, ys = [], []
    v_states, v_rets = [], []
    T = len(actions)
    R = torch.stack(rewards)
    for t in range(T):
        ok = alive[t]
        if not bool(ok.any()):
            continue
        died = ok & ~alive[t + 1]
        parent = clamp_state(raw[t][ok])
        if bank is None:
            child = clamp_state(raw[t + 1][ok])
        else:
            child = parent.clone()
            acts_ok = actions[t][ok]
            for a in range(4):
                rows = acts_ok == a
                program = bank.get(a)
                if program is not None and bool(rows.any()):
                    child[rows] = plant_executor(program, parent[rows])
        x = event_features(parent, child, actions[t][ok], psis[t][ok],
                           died=died[ok], soft=soft)
        xs.append(x)
        ys.append(rewards[t][ok])
        if t + args.horizon <= T:
            v_states.append(clamp_state(raw[t][ok]))
            v_rets.append(R[t:t + args.horizon, ok].sum(dim=0))
    X = torch.cat(xs)
    y = torch.cat(ys)
    S = torch.cat(v_states)
    ret = torch.cat(v_rets)
    return X, y, S, ret


def fit_typed(X, y, S, ret):
    w_r = ridge(X, y, args.ridge_lam)
    w_v = ridge(phi(S), ret, args.ridge_lam)
    return w_r, w_v


def fit_linear(raw, psis, actions, rewards, alive):
    """The F250 linear head on the same rollout (paired baseline)."""
    Ss, As, Rs, rets = [], [], [], []
    T = len(actions)
    R = torch.stack(rewards)
    for t in range(T):
        ok = alive[t]
        if not bool(ok.any()) or t + args.horizon > T:
            continue
        Ss.append(clamp_state(raw[t][ok]))
        As.append(actions[t][ok])
        Rs.append(rewards[t][ok])
        rets.append(R[t:t + args.horizon, ok].sum(dim=0))
    S = torch.cat(Ss); A = torch.cat(As)
    Rv = torch.cat(Rs); ret = torch.cat(rets)
    X = phi(S)
    w_r = torch.zeros(4, V_FEATS)
    for act in range(4):
        rows = A == act
        if int(rows.sum()) >= V_FEATS:
            w_r[act] = ridge(X[rows], Rv[rows], args.ridge_lam)
    w_v = ridge(X, ret, args.ridge_lam)
    return w_r, w_v


# ---- bank ----------------------------------------------------------

def collect_experience(config, seed, policy=None):
    """Tracked rollout recording states, psi snapshots, actions,
    rewards, avatar-vanish flags, and H-step returns."""
    B, T, H = args.data_batch, args.data_steps, args.horizon
    v = FamilyVerifier(config, batch_size=B, seed=seed)
    v.reset(seed=seed)
    g = torch.Generator().manual_seed(seed + 31337)
    tracker = Tracker(B)
    raw, psis, actions, rewards, alive = [], [], [], [], []
    code = tracker.encode(frame_cells(v.observation()))
    raw.append(code)
    psis.append(tracker.psi())
    alive.append(code[:, 0] < VALUES)
    for _ in range(T):
        state = clamp_state(code)
        if policy is None:
            action = torch.randint(0, 4, (B,), generator=g)
        else:
            action = policy(state, tracker.psi())
        actions.append(action)
        rewards.append(v.step(action).reward)
        code = tracker.encode(frame_cells(v.observation()))
        raw.append(code)
        psis.append(tracker.psi())
        alive.append(code[:, 0] < VALUES)
    return raw, psis, actions, rewards, alive


def build_head_data(raw, psis, actions, rewards, alive, soft=False,
                    bank=None):
    """Event-feature rows from a recorded rollout.  A row qualifies
    if the avatar was present at t; death at t+1 flows into the died
    flag (attribution).  With bank set, the child state is the
    BANK-ROLLED prediction from the parent for the taken action --
    model-consistent training (F253): the head is fit in the exact
    representation the planner will hand it."""
    xs, ys = [], []
    v_states, v_rets = [], []
    T = len(actions)
    R = torch.stack(rewards)
    for t in range(T):
        ok = alive[t]
        if not bool(ok.any()):
            continue
        died = ok & ~alive[t + 1]
        parent = clamp_state(raw[t][ok])
        if bank is None:
            child = clamp_state(raw[t + 1][ok])
        else:
            child = parent.clone()
            acts_ok = actions[t][ok]
            for a in range(4):
                rows = acts_ok == a
                program = bank.get(a)
                if program is not None and bool(rows.any()):
                    child[rows] = plant_executor(program, parent[rows])
        x = event_features(parent, child, actions[t][ok], psis[t][ok],
                           died=died[ok], soft=soft)
        xs.append(x)
        ys.append(rewards[t][ok])
        if t + args.horizon <= T:
            v_states.append(clamp_state(raw[t][ok]))
            v_rets.append(R[t:t + args.horizon, ok].sum(dim=0))
    X = torch.cat(xs)
    y = torch.cat(ys)
    S = torch.cat(v_states)
    ret = torch.cat(v_rets)
    return X, y, S, ret


def fit_typed(X, y, S, ret):
    w_r = ridge(X, y, args.ridge_lam)
    w_v = ridge(phi(S), ret, args.ridge_lam)
    return w_r, w_v


def fit_typed(X, y, S, ret):
    w_r = ridge(X, y, args.ridge_lam)
    w_v = ridge(phi(S), ret, args.ridge_lam)
    return w_r, w_v


def fit_linear(raw, psis, actions, rewards, alive):
    """The F250 linear head on the same rollout (paired baseline)."""
    Ss, As, Rs, rets = [], [], [], []
    T = len(actions)
    R = torch.stack(rewards)
    for t in range(T):
        ok = alive[t]
        if not bool(ok.any()) or t + args.horizon > T:
            continue
        Ss.append(clamp_state(raw[t][ok]))
        As.append(actions[t][ok])
        Rs.append(rewards[t][ok])
        rets.append(R[t:t + args.horizon, ok].sum(dim=0))
    S = torch.cat(Ss); A = torch.cat(As)
    Rv = torch.cat(Rs); ret = torch.cat(rets)
    X = phi(S)
    w_r = torch.zeros(4, V_FEATS)
    for act in range(4):
        rows = A == act
        if int(rows.sum()) >= V_FEATS:
            w_r[act] = ridge(X[rows], Rv[rows], args.ridge_lam)
    w_v = ridge(X, ret, args.ridge_lam)
    return w_r, w_v



# ---- grid library: class binding only (F267) ---------------------
TRAIN_WORLDS = [
    ("collect1", FamilyConfig(collect=1)),
    ("avoid1_collect1", FamilyConfig(avoid=1, collect=1)),
    ("pursue1", FamilyConfig(pursue=1)),
    ("intercept1", FamilyConfig(intercept=1)),
    ("collect1_resource1", FamilyConfig(collect=1, resource=1)),
    ("delayed3", FamilyConfig(delayed=3)),
]

report = {"seed": args.seed}
ev_psi, ev_y = [], []
for name, config in TRAIN_WORLDS:
    rollout = collect_experience(config, args.seed * 53)
    raw, psis, actions, rewards, alive = rollout
    for t in range(len(actions)):
        ok = alive[t]
        if not bool(ok.any()):
            continue
        died = ok & ~alive[t + 1]
        parent = clamp_state(raw[t])
        child = clamp_state(raw[t + 1])
        for g_ix, (_pl, base) in enumerate(TRACKED):
            d_now = ((child[:, 0] - child[:, base]).abs()
                     + (child[:, 1] - child[:, base + 1]).abs())
            hit_prev = ((child[:, 0] == parent[:, base])
                        & (child[:, 1] == parent[:, base + 1]))
            contact = ok & ((d_now <= 1) | hit_prev)
            d_par = ((parent[:, 0] - parent[:, base]).abs()
                     + (parent[:, 1] - parent[:, base + 1]).abs())
            contact = torch.where(died, ok & (d_par <= 2), contact)
            if bool(contact.any()):
                ev_psi.append(psis[t][contact, g_ix])
                ev_y.append(rewards[t][contact])
w_cl = ridge(torch.cat(ev_psi), torch.cat(ev_y), args.ridge_lam)
report["results"] = {}
print(f"class binding fit ({sum(x.shape[0] for x in ev_psi)} events)",
      flush=True)



# ---- ring substrate ----------------------------------------------
RN = 12


def ring_d(i, j):
    a = abs(int(i) - int(j))
    return min(a, RN - a)


class RingVerifier:
    def __init__(self, collect, avoid, batch_size, seed):
        self.collect, self.avoid = collect, avoid
        self.batch_size = batch_size
        self._gen = torch.Generator().manual_seed(seed)

    def reset(self, seed=None):
        if seed is not None:
            self._gen.manual_seed(int(seed))
        B = self.batch_size
        self.agent = torch.randint(0, RN, (B,), generator=self._gen)
        self.alive = torch.ones(B, dtype=torch.bool)
        self.food = (self.agent + torch.randint(
            1, RN, (B,), generator=self._gen)) % RN
        self.hazard = (self.agent + torch.randint(
            1, RN, (B,), generator=self._gen)) % RN

    def step(self, actions):
        B = self.batch_size
        delta = torch.where(actions == 0, -1, 1)
        self.agent = torch.where(self.alive,
                                 (self.agent + delta) % RN,
                                 self.agent)
        reward = torch.zeros(B)
        if self.collect:
            got = (self.agent == self.food) & self.alive
            reward = reward + got.float()
            fresh = (self.agent + torch.randint(
                1, RN, (B,), generator=self._gen)) % RN
            self.food = torch.where(got, fresh, self.food)
        if self.avoid:
            hop = torch.randint(0, 2, (B,), generator=self._gen) * 2 - 1
            self.hazard = (self.hazard + hop) % RN
            hit = (self.hazard == self.agent) & self.alive
            reward = reward - hit.float()
            self.alive = self.alive & ~hit
        return reward


class RingTracker:
    def __init__(self, batch):
        self.steps = torch.zeros(batch, 3)
        self.move = torch.zeros(batch, 3)
        self.closer = torch.zeros(batch, 3)
        self.prev = [[None, None] for _ in range(batch)]

    def psi(self):
        B = len(self.prev)
        steps = self.steps.clamp(min=1)
        out = torch.zeros(B, 3, PSI_DIM)
        out[:, :, 0] = 1.0
        out[:, 0, 1] = 1.0
        out[:, 1, 2] = 1.0
        out[:, :, 3] = self.move / steps
        out[:, :, 4] = self.closer / steps
        return out

    def observe(self, v):
        B = v.batch_size
        for b in range(B):
            if not bool(v.alive[b]):
                self.prev[b] = [None, None]
                continue
            a = int(v.agent[b])
            for t_ix, node in ((0, int(v.food[b])),
                               (1, int(v.hazard[b]))):
                if t_ix == 1 and not v.avoid:
                    continue
                p = self.prev[b][t_ix]
                if p is not None:
                    d_move = ring_d(p, node)
                    if d_move <= 2:
                        self.steps[b, t_ix] += 1
                        self.move[b, t_ix] += d_move
                        if ring_d(a, node) < ring_d(a, p):
                            self.closer[b, t_ix] += 1
                self.prev[b][t_ix] = node


def play_ring(collect, avoid, mode, w, seed):
    E, steps = args.episodes, args.steps
    v = RingVerifier(collect, avoid, E, seed)
    v.reset(seed=seed)
    g = torch.Generator().manual_seed(seed + 4242)
    tracker = RingTracker(E)
    tracker.observe(v)
    total = torch.zeros(E)
    for _ in range(steps):
        psi_now = tracker.psi()
        if mode == "random":
            action = torch.randint(0, 2, (E,), generator=g)
        elif mode == "gg":
            action = torch.zeros(E, dtype=torch.long)
            for b in range(E):
                a = int(v.agent[b])
                action[b] = min(
                    (0, 1), key=lambda p: ring_d(
                        (a + (-1 if p == 0 else 1)) % RN,
                        int(v.food[b])))
        elif mode == "valued_nav":
            action = torch.zeros(E, dtype=torch.long)
            best = torch.full((E,), -1e9)
            for p in (0, 1):
                score = torch.zeros(E)
                for b in range(E):
                    if not bool(v.alive[b]):
                        continue
                    a1 = int(v.agent[b])
                    a2 = (a1 + (-1 if p == 0 else 1)) % RN
                    ents = [(0, int(v.food[b]))]
                    if avoid:
                        ents.append((1, int(v.hazard[b])))
                    for g_, node in ents:
                        grad = ring_d(a1, node) - ring_d(a2, node)
                        val = float(psi_now[b, g_] @ w)
                        if val < 0:
                            # threat value is LOCAL (contact-driven):
                            # proximity-weight negative terms (F233's
                            # guard insight, made value-directed)
                            val *= max(0.0,
                                       (3.0 - ring_d(a1, node)) / 3.0)
                        score[b] += val * grad
                take = score > best
                best = torch.where(take, score, best)
                action = torch.where(
                    take, torch.full((E,), p, dtype=torch.long),
                    action)
        else:  # class head
            action = torch.zeros(E, dtype=torch.long)
            best = torch.full((E,), -1e9)
            for p in (0, 1):
                score = torch.zeros(E)
                for b in range(E):
                    if not bool(v.alive[b]):
                        continue
                    a2 = (int(v.agent[b])
                          + (-1 if p == 0 else 1)) % RN
                    ents = [(0, int(v.food[b]))]
                    if avoid:
                        ents.append((1, int(v.hazard[b])))
                    for g_, node in ents:
                        d = ring_d(a2, node)
                        k = max(0.0, (2.0 - d) / 2.0)
                        if k > 0:
                            score[b] += k * float(psi_now[b, g_] @ w)
                take = score > best
                best = torch.where(take, score, best)
                action = torch.where(
                    take, torch.full((E,), p, dtype=torch.long),
                    action)
        total += v.step(action)
        tracker.observe(v)
    return float(total.mean())


def fit_ring_native(collect, avoid, seed):
    """Ring-native class binding: (psi, reward) at co-location
    events under random play -- the native ceiling for this head."""
    B, T = 96, 40
    v = RingVerifier(collect, avoid, B, seed)
    v.reset(seed=seed)
    g = torch.Generator().manual_seed(seed + 8)
    tracker = RingTracker(B)
    tracker.observe(v)
    xs, ys = [], []
    for _ in range(T):
        psi_now = tracker.psi()
        pre_alive = v.alive.clone()
        pre_food = v.food.clone()
        pre_haz = v.hazard.clone()
        action = torch.randint(0, 2, (B,), generator=g)
        r = v.step(action)
        for b in range(B):
            if not bool(pre_alive[b]):
                continue
            a = int(v.agent[b])
            for g_, node in ((0, int(pre_food[b])),
                             (1, int(pre_haz[b]))):
                if g_ == 1 and not avoid:
                    continue
                if ring_d(a, node) <= 1:
                    xs.append(psi_now[b, g_])
                    ys.append(float(r[b]))
        tracker.observe(v)
    X = torch.stack(xs); y = torch.tensor(ys)
    return ridge(X, y, args.ridge_lam)


for wname, collect, avoid in (("rcollect1", 1, 0),
                              ("ravoid1_collect1", 1, 1)):
    row = {"random": play_ring(collect, avoid, "random", None,
                               args.seed * 977),
           "gg_ring": play_ring(collect, avoid, "gg", None,
                                args.seed * 977),
           "class_zeroshot": play_ring(collect, avoid, "class", w_cl,
                                       args.seed * 977)}
    w_nat = fit_ring_native(collect, avoid, args.seed * 61)
    row["native_ceiling"] = play_ring(collect, avoid, "class", w_nat,
                                      args.seed * 977)
    row["valued_nav"] = play_ring(collect, avoid, "valued_nav", w_cl,
                                  args.seed * 977)
    permn = torch.randperm(PSI_DIM,
                           generator=torch.Generator().manual_seed(
                               args.seed + 7070))
    row["valued_nav_shuf"] = play_ring(collect, avoid, "valued_nav",
                                       w_cl[permn], args.seed * 977)
    permc = torch.randperm(PSI_DIM,
                           generator=torch.Generator().manual_seed(
                               args.seed + 9090))
    row["shuffled_class"] = play_ring(collect, avoid, "class",
                                      w_cl[permc], args.seed * 977)
    report["results"][wname] = row
    print(f"  {wname:<18} " + "  ".join(
        f"{k} {v:+.3f}" for k, v in row.items()), flush=True)

print(json.dumps(report, indent=2))
if args.json:
    with open(args.json, "w") as handle:
        json.dump(report, handle, indent=2)

"""SEEDED LEARNING: transfer as ACCELERATION -- the founding
sentence, measured cross-substrate (F269).

F267 gave reliable zero-shot seeding (class binding, grid->graph);
F268 showed control wants the rich native head. The completion is
the founding objective's literal claim: experience gathered UNDER
the seeded policy teaches the native head faster than random
experience at equal budget. World: gavoid1_collect1 (zero-shot
+2.87, native ceiling +3.91 -- real headroom).

Per budget N (rollout steps at batch 96): collect graph data under
(a) the random policy, (b) the class-seeded policy; fit the native
pooled graph head on each; evaluate as policies.

Registered predictions (before any run):
  P1 acceleration: seeded-data head >= cold-data head + 0.3 at the
     smallest budget on >= 5/6 seeds.
  P2 convergence: the seeded/cold gap shrinks as N grows (both
     approach the native ceiling).
  P3 the seeded-data curve reaches 90% of the ceiling with <= 1/3
     of the samples the cold curve needs (reported as found).
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



# ---- graph substrate side ----------------------------------------

def bfs_dists(edges_row):
    """All-pairs BFS distances for one row's digraph, [N, N]."""
    INF = 99
    D = torch.full((NODES, NODES), INF, dtype=torch.long)
    for s in range(NODES):
        D[s, s] = 0
        frontier = [s]
        d = 0
        while frontier:
            d += 1
            nxt = []
            for u in frontier:
                for p in range(PORTS):
                    v = int(edges_row[u, p])
                    if D[s, v] > d:
                        D[s, v] = d
                        nxt.append(v)
            frontier = nxt
    return D


class GraphTracker:
    """Identity tracking + signatures on the graph: entity 0 = food
    (plane 1), entity 1 = hazard (plane 2); entity 2 absent."""

    def __init__(self, batch):
        self.node = [[None, None] for _ in range(batch)]
        self.steps = torch.zeros(batch, 3)
        self.move = torch.zeros(batch, 3)
        self.closer = torch.zeros(batch, 3)

    def psi(self):
        B = len(self.node)
        steps = self.steps.clamp(min=1)
        out = torch.zeros(B, 3, PSI_DIM)
        out[:, :, 0] = 1.0
        out[:, 0, 1] = 1.0
        out[:, 1, 2] = 1.0
        out[:, :, 3] = self.move / steps
        out[:, :, 4] = self.closer / steps
        return out

    def observe(self, v):
        """Returns (agent [B], nodes [B,2], dists [B,2] BFS to
        avatar, alive [B]); updates signatures."""
        B = v.batch_size
        obs = v.observation()
        agent = torch.full((B,), -1, dtype=torch.long)
        nodes = torch.full((B, 2), -1, dtype=torch.long)
        dists = torch.full((B, 2), 9, dtype=torch.long)
        alive = torch.zeros(B, dtype=torch.bool)
        for b in range(B):
            D = self.D[b]
            av = obs[b, 0].nonzero().flatten()
            if not len(av):
                self.node[b] = [None, None]
                continue
            alive[b] = True
            a = int(av[0])
            agent[b] = a
            for t_ix, plane in ((0, 1), (1, 2)):
                cells = obs[b, plane].nonzero().flatten().tolist()
                prev = self.node[b][t_ix]
                pick = None
                if prev is not None and cells:
                    near = min(cells, key=lambda z: int(D[prev, z]))
                    if int(D[prev, near]) <= args.match_dist:
                        pick = near
                if pick is None and cells:
                    pick = min(cells, key=lambda z: int(D[a, z]))
                if prev is not None and pick is not None:
                    d_move = int(D[prev, pick])
                    if d_move <= args.match_dist:
                        self.steps[b, t_ix] += 1
                        self.move[b, t_ix] += d_move
                        if int(D[a, pick]) < int(D[a, prev]):
                            self.closer[b, t_ix] += 1
                self.node[b][t_ix] = pick
                if pick is not None:
                    nodes[b, t_ix] = pick
                    dists[b, t_ix] = int(D[a, pick])
        return agent, nodes, dists, alive


def graph_event_features(D_all, agent_p, nodes_p, agent_c, nodes_c,
                         actions, psi):
    """Same R_FEATS layout as the grid head: per group, events
    [contact-only, teleport, both, boundary=0] x psi; groups 0-1
    live, group 2 inert."""
    N = agent_p.shape[0]
    feats = [torch.ones(N, 1)]
    for g in range(3):
        ev = torch.zeros(N, EVENTS)
        if g < 2:
            for b in range(N):
                if agent_c[b] < 0:
                    continue
                D = D_all[b]
                ep, ec_ = int(nodes_p[b, g]), int(nodes_c[b, g])
                if ec_ >= 0:
                    # substrate-calibrated kernel: the graph's BFS
                    # diameter is ~2, so "contact" sharpens to
                    # co-location-or-one-hop (metric normalization)
                    d_now = int(D[int(agent_c[b]), ec_])
                    contact = max(0.0, (1.5 - d_now) / 1.5)
                    if ep >= 0 and int(agent_c[b]) == ep:
                        contact = 1.0
                    tele = 1.0 if (ep >= 0
                                   and int(D[ep, ec_]) > 2) else 0.0
                    ev[b, 0] = contact * (1 - tele)
                    ev[b, 1] = tele
                    ev[b, 2] = contact * tele
        feats.append((ev.unsqueeze(2)
                      * psi[:, g].unsqueeze(1)).reshape(
                          N, EVENTS * PSI_DIM))
    feats.append(torch.nn.functional.one_hot(actions, 4).float())
    return torch.cat(feats, dim=1)


def graph_setup(v):
    return [bfs_dists(v.edges[b]) for b in range(v.batch_size)]


def collect_graph(config, seed):
    B, T = args.data_batch // 2, args.data_steps
    v = GraphVerifier(config, batch_size=B, seed=seed)
    v.reset(seed=seed)
    g = torch.Generator().manual_seed(seed + 31337)
    tracker = GraphTracker(B)
    tracker.D = graph_setup(v)
    xs, ys = [], []
    ag_p, no_p, _, alive_p = tracker.observe(v)
    psi_p = tracker.psi()
    for _ in range(T):
        action = torch.randint(0, PORTS, (B,), generator=g)
        r = v.step(action).reward
        ag_c, no_c, _, alive_c = tracker.observe(v)
        ok = alive_p
        if bool(ok.any()):
            x = graph_event_features(
                [tracker.D[b] for b in range(B)],
                ag_p, no_p, ag_c, no_c, action, psi_p)
            xs.append(x[ok])
            ys.append(r[ok])
        ag_p, no_p, alive_p, psi_p = ag_c, no_c, alive_c, tracker.psi()
    return torch.cat(xs), torch.cat(ys)


def play_graph(config, mode, w_r, seed):
    E, steps = args.episodes, args.steps
    v = GraphVerifier(config, batch_size=E, seed=seed)
    v.reset(seed=seed)
    g = torch.Generator().manual_seed(seed + 4242)
    tracker = GraphTracker(E)
    tracker.D = graph_setup(v)
    total = torch.zeros(E)
    ag, no, di, alive = tracker.observe(v)
    psi_now = tracker.psi()
    for _ in range(steps):
        if mode == "random":
            action = torch.randint(0, PORTS, (E,), generator=g)
        elif mode == "gg_bfs":
            action = torch.zeros(E, dtype=torch.long)
            for b in range(E):
                if ag[b] < 0 or no[b, 0] < 0:
                    continue
                D = tracker.D[b]
                action[b] = min(
                    range(PORTS),
                    key=lambda p: int(D[int(v.edges[b, ag[b], p]),
                                        int(no[b, 0])]))
        elif mode == "class":
            action = torch.zeros(E, dtype=torch.long)
            best = torch.full((E,), -1e9)
            for p in range(PORTS):
                score = torch.zeros(E)
                for b in range(E):
                    if ag[b] < 0:
                        continue
                    D = tracker.D[b]
                    ac_ = int(v.edges[b, ag[b], p])
                    for g in range(2):
                        if no[b, g] < 0:
                            continue
                        d = int(D[ac_, int(no[b, g])])
                        k = max(0.0, (1.5 - d) / 1.5)
                        if k > 0:
                            score[b] += k * float(
                                psi_now[b, g] @ w_r)
                take = score > best
                best = torch.where(take, score, best)
                action = torch.where(
                    take, torch.full((E,), p, dtype=torch.long),
                    action)
        else:  # typed head: score each port by predicted events
            action = torch.zeros(E, dtype=torch.long)
            best = torch.full((E,), -1e9)
            for p in range(PORTS):
                ag_c = torch.tensor([int(v.edges[b, ag[b], p])
                                     if ag[b] >= 0 else -1
                                     for b in range(E)])
                x = graph_event_features(
                    tracker.D, ag, no, ag_c, no,
                    torch.full((E,), p, dtype=torch.long), psi_now)
                r = x @ w_r
                take = r > best
                best = torch.where(take, r, best)
                action = torch.where(
                    take, torch.full((E,), p, dtype=torch.long),
                    action)
        total += v.step(action).reward
        ag, no, di, alive = tracker.observe(v)
        psi_now = tracker.psi()
    return float(total.mean())



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
print(f"class binding fit ({sum(x.shape[0] for x in ev_psi)} events)",
      flush=True)


def collect_graph_policy(config, seed, steps, policy_w):
    """Graph experience under the class-seeded policy (policy_w set)
    or the random policy (None); returns pooled-head rows."""
    B = 96
    v = GraphVerifier(config, batch_size=B, seed=seed)
    v.reset(seed=seed)
    g = torch.Generator().manual_seed(seed + 31337)
    tracker = GraphTracker(B)
    tracker.D = graph_setup(v)
    xs, ys = [], []
    ag_p, no_p, _, alive_p = tracker.observe(v)
    psi_p = tracker.psi()
    for _ in range(steps):
        if policy_w is None:
            action = torch.randint(0, PORTS, (B,), generator=g)
        else:
            action = torch.zeros(B, dtype=torch.long)
            best = torch.full((B,), -1e9)
            for p_ in range(PORTS):
                score = torch.zeros(B)
                for b in range(B):
                    if ag_p[b] < 0:
                        continue
                    D = tracker.D[b]
                    ac_ = int(v.edges[b, ag_p[b], p_])
                    for g_ in range(2):
                        if no_p[b, g_] < 0:
                            continue
                        d = int(D[ac_, int(no_p[b, g_])])
                        k = max(0.0, (1.5 - d) / 1.5)
                        if k > 0:
                            score[b] += k * float(
                                psi_p[b, g_] @ policy_w)
                take = score > best
                best = torch.where(take, score, best)
                action = torch.where(
                    take, torch.full((B,), p_, dtype=torch.long),
                    action)
        r = v.step(action).reward
        ag_c, no_c, _, alive_c = tracker.observe(v)
        ok = alive_p
        if bool(ok.any()):
            x = graph_event_features(
                [tracker.D[b] for b in range(B)],
                ag_p, no_p, ag_c, no_c, action, psi_p)
            xs.append(x[ok]); ys.append(r[ok])
        ag_p, no_p, alive_p, psi_p = ag_c, no_c, alive_c, tracker.psi()
    return torch.cat(xs), torch.cat(ys)


config = GraphConfig(collect=1, avoid=1)
row = {"random": play_graph(config, "random", None, args.seed * 977),
       "class_zeroshot": play_graph(config, "class", w_cl,
                                    args.seed * 977)}
Xc, yc = collect_graph_policy(config, args.seed * 61, 96, None)
w_ceil = ridge(Xc, yc, args.ridge_lam)
row["native_ceiling"] = play_graph(config, "typed", w_ceil,
                                   args.seed * 977)
for label, steps in (("n2", 2), ("n6", 6), ("n18", 18)):
    Xr, yr = collect_graph_policy(config, args.seed * 67, steps, None)
    w_r_ = ridge(Xr, yr, args.ridge_lam)
    row[f"cold_{label}"] = play_graph(config, "typed", w_r_,
                                      args.seed * 977)
    Xs, ys_ = collect_graph_policy(config, args.seed * 67, steps,
                                   w_cl)
    w_s_ = ridge(Xs, ys_, args.ridge_lam)
    row[f"seeded_{label}"] = play_graph(config, "typed", w_s_,
                                        args.seed * 977)
report["results"] = {"gavoid1_collect1": row}
print("  " + "  ".join(f"{k} {v:+.3f}" for k, v in row.items()),
      flush=True)

print(json.dumps(report, indent=2))
if args.json:
    with open(args.json, "w") as handle:
        json.dump(report, handle, indent=2)

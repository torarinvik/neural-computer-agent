"""COMPOUND BINDING: per-plane valued trackers compose (F239).

F238's open frontier: resource1_deceptive1 defeats both warm transfer
and cold discovery (~+0.2 each). The slot semantics say why: the
resource schedule's first leg targets NEAREST PLANE-2, and under
`avoid` that plane holds both the fuel and the threat -- the bait
aliasing, replayed on the other plane. The owned cure generalizes:
VALUE-BIND on BOTH planes. A plane-2 value fit separates resource
(contact reward 0) from hazard (-1) trivially; the compositional form
is a schedule of valued legs REACH(valued plane-2) -> REACH(valued
plane-1) under an encoder whose slots 4/5 and 6/7 hold the
argmax-value cells of each plane. No new grammar.

Arms: baseline = the F238 warm library; valued2x = + per-plane valued
forms. Registered: (1) resource1_deceptive1 >= +0.4 (from ~+0.2);
(2) resource1_avoid1 >= +0.3 (from cold's +0.096); (3) controls
resource1 / deceptive1 unchanged.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json

import torch

from experiments.games_amodal.fast_family import FastFamilyVerifier
from experiments.games_amodal.fast_stack import (
    score_goals, score_guards, score_schedules)
from experiments.games_amodal.game_family import FamilyConfig, FamilyVerifier

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=69316)
parser.add_argument("--dim", type=int, default=128)
parser.add_argument("--interpreter-updates", type=int, default=40000)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--examples", type=int, default=32)
parser.add_argument("--observations", type=int, default=256)
parser.add_argument("--episodes", type=int, default=64)
parser.add_argument("--steps", type=int, default=12)
parser.add_argument("--search-episodes", type=int, default=48)
parser.add_argument("--search-steps", type=int, default=12)
parser.add_argument("--json", default="")
args = parser.parse_args()

torch.set_num_threads(1)
torch.manual_seed(args.seed)

SLOTS, VALUES = 8, 8
ABSENT = VALUES
HEIGHT = WIDTH = 8
PLANES = 3
PAR_OPS = ("NOOP", "INC", "DEC", "CINC", "CDEC", "COPY", "SINC", "SDEC")
MODULI = tuple(range(2, VALUES + 1))
NOOP = (0, 0, 0)
ROWS_IX = torch.arange(HEIGHT).view(-1, 1).expand(HEIGHT, WIDTH)
COLS_IX = torch.arange(WIDTH).view(1, -1).expand(HEIGHT, WIDTH)


def slot_write(state, s, op, j, m):
    name, mod = PAR_OPS[op], MODULI[m]
    col = state[:, s]
    if name == "NOOP":
        return col
    if name == "INC":
        return (col + 1) % mod
    if name == "DEC":
        return (col - 1) % mod
    if name == "SINC":
        return torch.clamp(col + 1, max=mod - 1)
    if name == "SDEC":
        return torch.clamp(col - 1, min=0)
    if name == "CINC":
        return torch.where(state[:, j] != 0, (col + 1) % mod, col)
    if name == "CDEC":
        return torch.where(state[:, j] != 0, (col - 1) % mod, col)
    if name == "COPY":
        return state[:, j]
    raise AssertionError(name)


def run_parallel(state, program):
    out = state.clone()
    for s in range(SLOTS):
        out[:, s] = slot_write(state, s, *program[s])
    return out


def per_slot_search(before, after):
    program = []
    for s in range(SLOTS):
        want = after[:, s]
        best, best_score = NOOP, -1.0
        for op in range(len(PAR_OPS)):
            for j in range(SLOTS):
                if j == s and PAR_OPS[op] in ("CINC", "CDEC", "COPY"):
                    continue
                for m in range(len(MODULI)):
                    score = float((slot_write(before, s, op, j, m) == want)
                                  .float().mean())
                    if score > best_score:
                        best, best_score = (op, j, m), score
                    if best_score >= 1.0:
                        break
                if best_score >= 1.0:
                    break
            if best_score >= 1.0:
                break
        program.append(best)
    return program


class Interpreter(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.load = torch.nn.Linear(SLOTS * VALUES, dim)
        self.slot = torch.nn.Embedding(SLOTS, dim)
        self.op = torch.nn.Embedding(len(PAR_OPS), dim)
        self.arg_j = torch.nn.Embedding(SLOTS, dim)
        self.arg_m = torch.nn.Embedding(len(MODULI), dim)
        self.step = torch.nn.Sequential(
            torch.nn.Linear(3 * dim, 2 * dim), torch.nn.ReLU(),
            torch.nn.Linear(2 * dim, 2 * dim), torch.nn.ReLU(),
            torch.nn.Linear(2 * dim, dim))
        self.norm = torch.nn.LayerNorm(dim)
        self.head = torch.nn.Linear(dim, SLOTS * VALUES)

    def forward(self, program, state):
        hot = torch.nn.functional.one_hot(
            state, VALUES).float().view(state.shape[0], -1)
        base = self.load(hot)
        latent = base
        for s in range(SLOTS):
            op, j, m = program[s]
            code = (self.slot(torch.tensor(s)) + self.op(torch.tensor(op))
                    + self.arg_j(torch.tensor(j))
                    + self.arg_m(torch.tensor(m))).unsqueeze(0).expand(
                        latent.shape[0], -1)
            latent = self.norm(latent + self.step(
                torch.cat([latent, base, code], dim=-1)))
        return self.head(latent).view(-1, SLOTS, VALUES)


def random_program(g):
    out = []
    for s in range(SLOTS):
        op = int(torch.randint(0, len(PAR_OPS), (1,), generator=g))
        j = int(torch.randint(0, SLOTS, (1,), generator=g))
        if j == s:
            j = (j + 1) % SLOTS
        out.append((op, j, int(torch.randint(0, len(MODULI), (1,),
                                             generator=g))))
    return out


interp = Interpreter(args.dim)
opt = torch.optim.AdamW(interp.parameters(), lr=args.lr, weight_decay=0.01)
gen = torch.Generator().manual_seed(args.seed * 104729)
for _ in range(args.interpreter_updates):
    prog = random_program(gen)
    st = torch.randint(0, VALUES, (args.batch_size, SLOTS), generator=gen)
    loss = torch.nn.functional.cross_entropy(
        interp(prog, st).reshape(-1, VALUES),
        run_parallel(st, prog).reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
for p in interp.parameters():
    p.requires_grad_(False)
check = torch.Generator().manual_seed(args.seed + 5551)
hits = rows = 0
for _ in range(32):
    prog = random_program(check)
    st = torch.randint(0, VALUES, (128, SLOTS), generator=check)
    with torch.no_grad():
        hits += int((interp(prog, st).argmax(-1)
                     == run_parallel(st, prog)).sum())
    rows += st.numel()
report = {"seed": args.seed, "interpreter_check": round(hits / rows, 4)}


def plant_executor(program, state):
    with torch.no_grad():
        return interp(program, state).argmax(-1)


def _kth_nearest(plane, ref_row, ref_col, k):
    mask = plane > 0
    d = ((ROWS_IX.unsqueeze(0) - ref_row.view(-1, 1, 1)).abs()
         + (COLS_IX.unsqueeze(0) - ref_col.view(-1, 1, 1)).abs())
    d = torch.where(mask, d, torch.full_like(d, 999))
    flat = d.reshape(d.shape[0], -1)
    order = flat.argsort(dim=1)
    idx = order[:, min(k, order.shape[1] - 1)]
    enough = mask.reshape(mask.shape[0], -1).sum(dim=1) > k
    row = torch.where(enough, idx // WIDTH, torch.full_like(idx, ABSENT))
    col = torch.where(enough, idx % WIDTH, torch.full_like(idx, ABSENT))
    return row, col


def _approaching(prev_plane, curr_plane, avatar_r, avatar_c):
    """Generic looming reduction: one-step optical flow (newly-occupied
    cells matched to their nearest newly-vacated origin), keeping the
    nearest cell whose move reduced its distance to the avatar."""
    batch = curr_plane.shape[0]
    rows = torch.full((batch,), ABSENT, dtype=torch.long)
    cols = torch.full((batch,), ABSENT, dtype=torch.long)
    for i in range(batch):
        if int(avatar_r[i]) >= VALUES:
            continue
        now = {(int(r), int(c)) for r, c in (curr_plane[i] > 0).nonzero()}
        was = {(int(r), int(c)) for r, c in (prev_plane[i] > 0).nonzero()}
        fresh, gone = now - was, was - now
        if not fresh or not gone:
            continue
        ar, ac = int(avatar_r[i]), int(avatar_c[i])
        best = None
        for r, c in fresh:
            o = min(gone, key=lambda g: abs(g[0] - r) + abs(g[1] - c))
            d_new = abs(r - ar) + abs(c - ac)
            d_old = abs(o[0] - ar) + abs(o[1] - ac)
            if d_new < d_old and (best is None or d_new < best[0]):
                best = (d_new, r, c)
        if best is not None:
            rows[i], cols[i] = best[1], best[2]
    return rows, cols


def make_enc(kind):
    """Slots 0-1 avatar, 2-3 nearest plane 1, 4-5 nearest plane 2.
    Slots 6/7: `second2` = second-nearest of plane 2 (F225);
    `approach2` = nearest approaching cell of plane 2 (temporal)."""
    def encoder(prev_screen, screen):
        frames = screen.view(-1, PLANES, HEIGHT, WIDTH)
        prior = prev_screen.view(-1, PLANES, HEIGHT, WIDTH)
        out = torch.full((frames.shape[0], SLOTS), ABSENT, dtype=torch.long)
        avatar = frames[:, 0].reshape(frames.shape[0], -1)
        present = avatar.max(dim=1).values > 0
        flat = avatar.argmax(dim=1)
        ar = torch.where(present, flat // WIDTH,
                         torch.full_like(flat, ABSENT))
        ac = torch.where(present, flat % WIDTH,
                         torch.full_like(flat, ABSENT))
        out[:, 0], out[:, 1] = ar, ac
        for plane, base in ((1, 2), (2, 4)):
            row, col = _kth_nearest(frames[:, plane], ar.clamp(max=VALUES-1),
                                    ac.clamp(max=VALUES-1), 0)
            out[:, base], out[:, base + 1] = row, col
        if kind == "second":
            row, col = _kth_nearest(frames[:, 2], ar.clamp(max=VALUES-1),
                                    ac.clamp(max=VALUES-1), 1)
        else:
            row, col = _approaching(prior[:, 2], frames[:, 2], ar, ac)
        out[:, 6], out[:, 7] = row, col
        return out
    return encoder


def _clear_nearest_slow(plane_t, plane_h, ar, ac, thresh):
    mask_t = plane_t > 0
    d_h = torch.full_like(plane_t, 999.0)
    hz = plane_h > 0
    rows_ix = ROWS_IX.unsqueeze(0).float()
    cols_ix = COLS_IX.unsqueeze(0).float()
    for i in range(plane_t.shape[0]):
        cells = hz[i].nonzero()
        if cells.numel():
            d = ((rows_ix[0].unsqueeze(-1) - cells[:, 0].float()).abs()
                 + (cols_ix[0].unsqueeze(-1) - cells[:, 1].float()).abs())
            d_h[i] = d.min(dim=-1).values
    ok = mask_t & (d_h >= thresh)
    d_av = ((ROWS_IX.unsqueeze(0) - ar.view(-1, 1, 1)).abs()
            + (COLS_IX.unsqueeze(0) - ac.view(-1, 1, 1)).abs())
    d_av = torch.where(ok, d_av, torch.full_like(d_av, 999))
    flat = d_av.reshape(d_av.shape[0], -1)
    best = flat.argmin(dim=1)
    found = ok.reshape(ok.shape[0], -1).any(dim=1) & (ar < VALUES)
    row = torch.where(found, best // WIDTH, torch.full_like(best, ABSENT))
    col = torch.where(found, best % WIDTH, torch.full_like(best, ABSENT))
    return row, col


def make_clear_enc(thresh):
    def encoder(prev_screen, screen):
        frames = screen.view(-1, PLANES, HEIGHT, WIDTH)
        out = torch.full((frames.shape[0], SLOTS), ABSENT,
                         dtype=torch.long)
        avatar = frames[:, 0].reshape(frames.shape[0], -1)
        present = avatar.max(dim=1).values > 0
        flat = avatar.argmax(dim=1)
        ar = torch.where(present, flat // WIDTH,
                         torch.full_like(flat, ABSENT))
        ac = torch.where(present, flat % WIDTH,
                         torch.full_like(flat, ABSENT))
        out[:, 0], out[:, 1] = ar, ac
        for plane, base in ((1, 2), (2, 4)):
            row, col = _kth_nearest(frames[:, plane],
                                    ar.clamp(max=VALUES - 1),
                                    ac.clamp(max=VALUES - 1), 0)
            out[:, base], out[:, base + 1] = row, col
        row, col = _clear_nearest_slow(frames[:, 1], frames[:, 2],
                                       ar, ac, thresh)
        out[:, 6], out[:, 7] = row, col
        return out
    return encoder


class EntityTable:
    """Persistent identity for static objects: cell occupancy IS the
    entity. Records each entity's relational context at spawn."""

    def __init__(self, plane=1, other=2):
        self.rows = None
        self.plane, self.other = plane, other

    def reset(self, batch):
        self.rows = [dict() for _ in range(batch)]

    def update(self, frames):
        batch = frames.shape[0]
        if self.rows is None or len(self.rows) != batch:
            self.reset(batch)
        for i in range(batch):
            cells = {(int(r), int(c)) for r, c in
                     (frames[i, self.plane] > 0).nonzero()}
            hz = [(int(r), int(c)) for r, c in
                  (frames[i, self.other] > 0).nonzero()]
            table = self.rows[i]
            for cell in list(table):
                if cell not in cells:
                    del table[cell]
            for cell in cells:
                if cell not in table:
                    if hz:
                        dm = min(abs(cell[0] - hr) + abs(cell[1] - hc)
                                 for hr, hc in hz)
                        dr = min(abs(cell[0] - hr) for hr, hc in hz)
                    else:
                        dm, dr = 8.0, 8.0
                    table[cell] = (float(dm), float(dr))


TRACKER = EntityTable(plane=1, other=2)
TRACKER2 = EntityTable(plane=2, other=1)


class AgentHistory:
    """Steps since the avatar last stood on a plane-2 cell, capped.
    Read from consecutive frames only -- generic self-history."""

    def __init__(self, horizon=4):
        self.horizon = horizon
        self.since = None

    def reset(self, batch):
        self.since = [self.horizon] * batch

    def update(self, prev_frames, frames):
        batch = frames.shape[0]
        if self.since is None or len(self.since) != batch:
            self.reset(batch)
        for i in range(batch):
            avatar = frames[i, 0].reshape(-1)
            if float(avatar.max()) <= 0:
                continue
            pos = int(avatar.argmax())
            r, c = pos // WIDTH, pos % WIDTH
            if float(prev_frames[i, 2, r, c]) > 0:
                self.since[i] = 0
            else:
                self.since[i] = min(self.horizon, self.since[i] + 1)

    def fueled(self, i):
        return 1.0 if (self.since is not None and i < len(self.since)
                       and self.since[i] < self.horizon) else 0.0


AGENT_HIST = AgentHistory()


def _phi_cells_plane(frames, ar, ac, plane, other, tracker):
    out = []
    for i in range(frames.shape[0]):
        cells = [(int(r), int(c)) for r, c in
                 (frames[i, plane] > 0).nonzero()]
        hz = [(int(r), int(c)) for r, c in
              (frames[i, other] > 0).nonzero()]
        row_out = []
        for (r, c) in cells:
            d_av = abs(r - int(ar[i])) + abs(c - int(ac[i]))
            if hz:
                dm = min(abs(r - hr) + abs(c - hc) for hr, hc in hz)
                dr = min(abs(r - hr) for hr, hc in hz)
            else:
                dm, dr = 8.0, 8.0
            spawn = (tracker.rows[i].get((r, c), (dm, dr))
                     if tracker.rows is not None
                     and i < len(tracker.rows) else (dm, dr))
            row_out.append(((r, c), (1.0, float(d_av), float(dm),
                                     float(dr), spawn[0], spawn[1],
                                     AGENT_HIST.fueled(i))))
        out.append(row_out)
    return out


def _phi_cells(frames, ar, ac):
    return _phi_cells_plane(frames, ar, ac, 1, 2, TRACKER)


def _phi_cells_legacy(frames, ar, ac):
    """Generic relational token per plane-1 set cell, per row:
    [1, d(o, avatar), d_manh(o, nearest plane-2), d_row(o, plane-2)].
    Returns (cells, phis) as python lists per row."""
    out = []
    for i in range(frames.shape[0]):
        cells = [(int(r), int(c)) for r, c in
                 (frames[i, 1] > 0).nonzero()]
        hz = [(int(r), int(c)) for r, c in (frames[i, 2] > 0).nonzero()]
        row_out = []
        for (r, c) in cells:
            d_av = abs(r - int(ar[i])) + abs(c - int(ac[i]))
            if hz:
                dm = min(abs(r - hr) + abs(c - hc) for hr, hc in hz)
                dr = min(abs(r - hr) for hr, hc in hz)
            else:
                dm, dr = 8.0, 8.0
            spawn = (TRACKER.rows[i].get((r, c), (dm, dr))
                     if TRACKER.rows is not None
                     and i < len(TRACKER.rows) else (dm, dr))
            row_out.append(((r, c), (1.0, float(d_av), float(dm),
                                     float(dr), spawn[0], spawn[1])))
        out.append(row_out)
    return out


def make_valued_enc(mode, w=None):
    def encoder(prev_screen, screen):
        frames = screen.view(-1, PLANES, HEIGHT, WIDTH)
        TRACKER.update(frames)
        out = torch.full((frames.shape[0], SLOTS), ABSENT,
                         dtype=torch.long)
        avatar = frames[:, 0].reshape(frames.shape[0], -1)
        present = avatar.max(dim=1).values > 0
        flat = avatar.argmax(dim=1)
        ar = torch.where(present, flat // WIDTH,
                         torch.full_like(flat, ABSENT))
        ac = torch.where(present, flat % WIDTH,
                         torch.full_like(flat, ABSENT))
        out[:, 0], out[:, 1] = ar, ac
        for plane, base in ((1, 2), (2, 4)):
            row, col = _kth_nearest(frames[:, plane],
                                    ar.clamp(max=VALUES - 1),
                                    ac.clamp(max=VALUES - 1), 0)
            out[:, base], out[:, base + 1] = row, col
        if mode == "true":
            v = HOOK["v"]
            for i in range(frames.shape[0]):
                if int(ar[i]) >= VALUES or v is None:
                    continue
                foods = v._food[i] if i < len(v._food) else []
                if foods:
                    best = min(foods, key=lambda f: abs(f[0] - int(ar[i]))
                               + abs(f[1] - int(ac[i])))
                    out[i, 6], out[i, 7] = best
        else:
            rows = _phi_cells(frames, ar, ac)
            for i, row_cells in enumerate(rows):
                if int(ar[i]) >= VALUES or not row_cells:
                    continue
                w_row = (w[1] if AGENT_HIST.fueled(i) > 0 else w[0])                     if isinstance(w, tuple) else w
                best = max(row_cells,
                           key=lambda cp: sum(a * b for a, b in
                                              zip(w_row, cp[1])))
                out[i, 6], out[i, 7] = best[0]
        return out
    return encoder


def make_valued2x_enc(w1, w2):
    """Slots 4/5 = argmax-value plane-2 cell; slots 6/7 = argmax-value
    plane-1 cell. Both entity trackers updated every frame."""
    def encoder(prev_screen, screen):
        frames = screen.view(-1, PLANES, HEIGHT, WIDTH)
        prior = prev_screen.view(-1, PLANES, HEIGHT, WIDTH)
        TRACKER.update(frames)
        TRACKER2.update(frames)
        AGENT_HIST.update(prior, frames)
        out = torch.full((frames.shape[0], SLOTS), ABSENT,
                         dtype=torch.long)
        avatar = frames[:, 0].reshape(frames.shape[0], -1)
        present = avatar.max(dim=1).values > 0
        flat = avatar.argmax(dim=1)
        ar = torch.where(present, flat // WIDTH,
                         torch.full_like(flat, ABSENT))
        ac = torch.where(present, flat % WIDTH,
                         torch.full_like(flat, ABSENT))
        out[:, 0], out[:, 1] = ar, ac
        row, col = _kth_nearest(frames[:, 1], ar.clamp(max=VALUES - 1),
                                ac.clamp(max=VALUES - 1), 0)
        out[:, 2], out[:, 3] = row, col
        for plane, other, w, tracker, base in (
                (2, 1, w2, TRACKER2, 4), (1, 2, w1, TRACKER, 6)):
            rows = _phi_cells_plane(frames, ar, ac, plane, other,
                                    tracker)
            for i, row_cells in enumerate(rows):
                if int(ar[i]) >= VALUES or not row_cells:
                    continue
                if plane == 1 and isinstance(w, tuple):
                    w_row = w[1] if AGENT_HIST.fueled(i) > 0 else w[0]
                else:
                    w_row = w
                best = max(row_cells,
                           key=lambda cp: sum(a * b for a, b in
                                              zip(w_row, cp[1])))
                out[i, base], out[i, base + 1] = best[0]
        return out
    return encoder


ENC_CANDIDATES = {"second2": make_enc("second"),
                  "approach2": make_enc("approach"),
                  "clear1_2": make_clear_enc(2),
                  "clear1_3": make_clear_enc(3)}
enc = ENC_CANDIDATES["second2"]           # rebound per world below


def goal_cost(state, reference, goal):
    if goal and isinstance(goal[0][0], int):
        goal = (goal,)
    total = None
    for (a0, a1), (b0, b1), sign in goal:
        reach = ((state[:, a0] - reference[:, b0]).abs()
                 + (state[:, a1] - reference[:, b1]).abs()).float()
        term = sign * reach
        total = term if total is None else total + term
    return total


def build_bank(config, seed, executor):
    """One random warmup step gives every temporal reduction a real
    (prev, curr) pair before the measured transition."""
    g = torch.Generator().manual_seed(seed + 999)

    def warmed(v, s):
        v.reset(seed=s)
        first = v.observation()
        v.step(torch.randint(0, 4, (args.observations,), generator=g))
        return first, v.observation()

    probe = FamilyVerifier(config, batch_size=args.observations,
                           seed=seed + 7)
    first, second = warmed(probe, seed + 7)
    used = (enc(first, second) < VALUES).float().mean(dim=0) >= 0.9
    bank = {}
    for act in range(4):
        v = FamilyVerifier(config, batch_size=args.observations,
                           seed=seed + act)
        first, second = warmed(v, seed + act)
        before = enc(first, second)
        v.step(torch.full((args.observations,), act, dtype=torch.long))
        after = enc(second, v.observation())
        keep = ((before[:, used] < VALUES).all(dim=1)
                & (after[:, used] < VALUES).all(dim=1))
        if int(keep.sum()) < 8:
            continue
        b = torch.where(before[keep] < VALUES, before[keep],
                        torch.zeros_like(before[keep]))
        a = torch.where(after[keep] < VALUES, after[keep],
                        torch.zeros_like(after[keep]))
        bank[act] = per_slot_search(b[:args.examples], a[:args.examples])
    return bank


def play(config, mode, bank, seed, goal, executor, episodes, steps):
    """`goal` is either a plain tuple-of-terms goal or a SCHEDULE
    ("schedule", [phase0_goal, phase1_goal]): each row advances its
    phase cyclically whenever the active goal's cost at the CURRENT
    reference is <= 0 (arrival)."""
    schedule = None
    guard = None
    if goal is not None and goal and goal[0] == "schedule":
        schedule = goal[1]
    elif goal is not None and goal and goal[0] == "guard":
        guard = goal[1:]  # (condition, goal_A, goal_B)
    v = FamilyVerifier(config, batch_size=episodes, seed=seed)
    v.reset(seed=seed)
    HOOK["v"] = v
    TRACKER.reset(episodes)
    TRACKER2.reset(episodes)
    AGENT_HIST.reset(episodes)
    g = torch.Generator().manual_seed(seed + 4242)
    total = torch.zeros(episodes)
    prev = v.observation()
    phase = torch.zeros(episodes, dtype=torch.long)
    prev_ref = None
    prev_cost = torch.full((episodes,), 1e9)
    for _ in range(steps):
        obs = v.observation()
        if mode == "random":
            action = torch.randint(0, 4, (episodes,), generator=g)
        else:
            reference = enc(prev, obs)
            reference = torch.where(reference < VALUES, reference,
                                    torch.zeros_like(reference))
            if guard is not None:
                (c0, c1), (d0, d1), theta = guard[0]
                gap = ((reference[:, c0] - reference[:, d0]).abs()
                       + (reference[:, c1] - reference[:, d1]).abs())
                take_a = gap >= theta
            if schedule is not None:
                for p in range(len(schedule)):
                    here = goal_cost(reference, reference, schedule[p])
                    if prev_ref is None:
                        consumed = torch.zeros_like(here,
                                                    dtype=torch.bool)
                    else:
                        jump = torch.zeros_like(here)
                        for _pa, (b0, b1), _sign in schedule[p]:
                            step_jump = (
                                (reference[:, b0]
                                 - prev_ref[:, b0]).abs().float()
                                + (reference[:, b1]
                                   - prev_ref[:, b1]).abs().float())
                            jump = torch.maximum(jump, step_jump)
                        consumed = (prev_cost <= 1.0) & (jump > 1.0)
                    done = (phase == p) & ((here <= 0) | consumed)
                    phase = torch.where(
                        done, torch.full_like(phase,
                                              (p + 1) % len(schedule)),
                        phase)
                active_cost = torch.zeros_like(reference[:, 0]).float()
                for p in range(len(schedule)):
                    rows = phase == p
                    if bool(rows.any()):
                        active_cost[rows] = goal_cost(
                            reference[rows], reference[rows], schedule[p])
                prev_cost = active_cost
                prev_ref = reference
            best, action = None, torch.zeros(episodes, dtype=torch.long)
            for act in range(4):
                if mode == "oracle":
                    shadow = copy.deepcopy(v)
                    shadow.step(torch.full((episodes,), act))
                    seen = enc(obs, shadow.observation())
                    state = torch.where(seen < VALUES, seen,
                                        torch.zeros_like(seen))
                else:
                    program = bank.get(act)
                    state = (reference if program is None
                             else executor(program, reference))
                if guard is not None:
                    cost = torch.zeros(episodes)
                    for branch, sub in ((True, guard[1]), (False, guard[2])):
                        rows = take_a if branch else ~take_a
                        if bool(rows.any()):
                            cost[rows] = goal_cost(
                                state[rows], reference[rows], sub)
                elif schedule is None:
                    cost = goal_cost(state, reference, goal)
                else:
                    cost = torch.zeros(episodes)
                    for p in range(len(schedule)):
                        rows = phase == p
                        if bool(rows.any()):
                            cost[rows] = goal_cost(
                                state[rows], reference[rows], schedule[p])
                if best is None:
                    best = cost.clone()
                else:
                    take = cost < best
                    best = torch.where(take, cost, best)
                    action = torch.where(
                        take, torch.full((episodes,), act), action)
        total += v.step(action).reward
        prev = obs
    return float(total.mean())


def usable_slots(config):
    v = FamilyVerifier(config, batch_size=args.observations,
                       seed=args.seed * 31)
    v.reset(seed=args.seed * 31)
    first = v.observation()
    g = torch.Generator().manual_seed(args.seed * 31 + 999)
    v.step(torch.randint(0, 4, (args.observations,), generator=g))
    code = enc(first, v.observation())
    present = (code < VALUES).float().mean(dim=0)
    return {s for s in range(SLOTS) if float(present[s]) >= 0.9}


class FastHookAdapter:
    """Presents the fast verifier's food tensor as ._food lists so the
    privileged valued encoder works on both selection and evaluation."""

    def __init__(self, fv):
        self._fv = fv

    @property
    def _food(self):
        food = self._fv.food
        return [[(int(r), int(c)) for r, c in food[i]]
                for i in range(food.shape[0])]


HOOK = {"v": None}   # the verifier currently rendering; privileged
                     # encoders may read its private item lists.

pairs = [q for q in itertools.permutations(range(SLOTS), 2)]


def choose_by_race(config, bank, label):
    """Batched race: all plain singles + all avatar-anchored 2-phase
    schedules, robust min over two selection streams on the fast
    verifier."""
    usable = usable_slots(config)
    singles = []
    for pa in pairs:
        for pb in pairs:
            if set(pa) & set(pb) or pa[0] > pa[1]:
                continue
            if not (set(pa) | set(pb)) <= usable:
                continue
            for sign in (1, -1):
                singles.append(((pa, pb, sign),))
    legs = [((0, 1), pb, 1) for pb in pairs
            if 0 not in pb and 1 not in pb and set(pb) <= usable]
    schedules = [((legs[i],), (legs[j],))
                 for i in range(len(legs)) for j in range(len(legs))
                 if i != j]
    if not singles:
        return None

    def factory(batch):
        fv = FastFamilyVerifier(config, batch_size=batch,
                                seed=args.seed * 31)
        HOOK["v"] = FastHookAdapter(fv)
        return fv

    if label in FAST_ENC:
        fenc = FAST_ENC[label]
    else:
        fenc = globals()["FAST_ENC_OVERRIDE"]

    def race(items, scorer):
        score = None
        for off in (1, 2):
            s = scorer(factory, items, bank, plant_executor, fenc,
                       args.search_episodes, args.search_steps,
                       args.seed * 977 + off)
            score = s if score is None else torch.minimum(score, s)
        return score

    single_scores = race(singles, score_goals)
    best_i = int(single_scores.argmax())
    best_goal, best_score = singles[best_i], float(single_scores.max())
    # F232: race sum-composites of the top-16 singles as well -- the
    # bait tax may be expressible as approach-food + avoid-hazard
    order = single_scores.argsort(descending=True)[:16].tolist()
    composites = [(singles[i][0], singles[j][0])
                  for i in order for j in order if i != j]
    if composites:
        comp_scores = race(composites, score_goals)
        if float(comp_scores.max()) > best_score:
            k = int(comp_scores.argmax())
            best_goal = composites[k]
            best_score = float(comp_scores.max())
    if schedules:
        sched_scores = race(schedules, score_schedules)
        if float(sched_scores.max()) > best_score:
            j = int(sched_scores.argmax())
            best_goal = ("schedule", list(schedules[j]))
            best_score = float(sched_scores.max())
    return best_goal, best_score


def fit_value_w_plane(config, seed, plane):
    """plane 1: class 1{r >= 0.5} (food vs bait/death). plane 2:
    class 1{r > -0.5} (safe touch vs fatal): resource vs hazard."""
    tracker = TRACKER if plane == 1 else TRACKER2
    other = 2 if plane == 1 else 1
    xs, rs = [], []
    g = torch.Generator().manual_seed(seed)
    for arm in range(10):
        v = FamilyVerifier(config, batch_size=128, seed=seed + arm * 71)
        v.reset(seed=seed + arm * 71)
        tracker.reset(128)
        AGENT_HIST.reset(128)
        prev_frames = v.observation().view(-1, PLANES, HEIGHT, WIDTH)
        for _ in range(12):
            frames = v.observation().view(-1, PLANES, HEIGHT, WIDTH)
            AGENT_HIST.update(prev_frames, frames)
            prev_frames = frames
            avatar = frames[:, 0].reshape(frames.shape[0], -1)
            flat = avatar.argmax(dim=1)
            ar, ac = flat // WIDTH, flat % WIDTH
            tracker.update(frames)
            cell_rows = _phi_cells_plane(frames, ar, ac, plane, other,
                                         tracker)
            before_positions = [dict(rc) for rc in cell_rows]
            actions = torch.randint(0, 4, (128,), generator=g)
            out = v.step(actions)
            for i in range(128):
                target = v._avatar[i]
                r = float(out.reward[i])
                died = not bool(out.alive[i])
                if target in before_positions[i]:
                    xs.append(before_positions[i][target])
                    good = (r >= 0.5) if plane == 1 else (r > -0.5
                                                          and not died)
                    rs.append(1.0 if good else 0.0)
                elif plane == 2 and died and before_positions[i]:
                    # death attribution: the nearest plane-2 entity to
                    # where the avatar died carries the bad label --
                    # moving threats kill without being stepped on.
                    killer = min(
                        before_positions[i],
                        key=lambda c: abs(c[0] - target[0])
                        + abs(c[1] - target[1]))
                    xs.append(before_positions[i][killer])
                    rs.append(0.0)
    if len(xs) < 16:
        return None, 0.0
    X = torch.tensor(xs)
    r = torch.tensor(rs)
    half = len(xs) // 2

    def fit(X_, r_):
        return torch.linalg.solve(X_.T @ X_ + torch.eye(X_.shape[1]),
                                  X_.T @ r_)

    if plane == 1:
        # state-conditioned value: stratify by the fueled bit (last
        # phi feature). Food's value is only informative when fueled;
        # pooling corrupts the labels under resource gating.
        flag = X[:, -1] > 0.5
        strata = []
        for mask in (flag, ~flag):
            if int(mask.sum()) >= 16:
                strata.append(fit(X[mask], r[mask]))
            else:
                strata.append(None)
        pooled = fit(X, r)
        w_fueled = strata[0] if strata[0] is not None else pooled
        w_unfueled = strata[1] if strata[1] is not None else pooled
    w = fit(X[:half], r[:half])
    vh = X[half:] @ w
    rh = r[half:]
    wins = ties = tot = 0
    for i in range(len(rh)):
        for j in range(i + 1, len(rh)):
            if rh[i] == rh[j]:
                continue
            tot += 1
            better = i if rh[i] > rh[j] else j
            worse = j if better == i else i
            if vh[better] > vh[worse]:
                wins += 1
            elif vh[better] == vh[worse]:
                ties += 1
    acc = (wins + 0.5 * ties) / tot if tot else 0.0
    if plane == 1:
        return (w_unfueled.tolist(), w_fueled.tolist()), round(acc, 4)
    return fit(X, r).tolist(), round(acc, 4)


def fit_value_w(config, seed):
    """Ridge fit of contact reward on phi from random rollouts;
    returns (w, held_ranking_accuracy)."""
    xs, rs = [], []
    g = torch.Generator().manual_seed(seed)
    for arm in range(10):
        v = FamilyVerifier(config, batch_size=128, seed=seed + arm * 71)
        v.reset(seed=seed + arm * 71)
        TRACKER.reset(128)
        for _ in range(12):
            frames = v.observation().view(-1, PLANES, HEIGHT, WIDTH)
            avatar = frames[:, 0].reshape(frames.shape[0], -1)
            flat = avatar.argmax(dim=1)
            ar, ac = flat // WIDTH, flat % WIDTH
            TRACKER.update(frames)
            cell_rows = _phi_cells(frames, ar, ac)
            before_positions = [dict(rc) for rc in cell_rows]
            actions = torch.randint(0, 4, (128,), generator=g)
            out = v.step(actions)
            for i in range(128):
                target = v._avatar[i]
                if target in before_positions[i]:
                    xs.append(before_positions[i][target])
                    # class target: 1 = high-value contact (food),
                    # 0 = low (bait) or catastrophic (death same step)
                    rs.append(1.0 if float(out.reward[i]) >= 0.5
                              else 0.0)
    if len(xs) < 16:
        return None, 0.0
    X = torch.tensor(xs)
    r = torch.tensor(rs)
    half = len(xs) // 2
    def fit(X_, r_):
        return torch.linalg.solve(X_.T @ X_ + torch.eye(X_.shape[1]),
                                  X_.T @ r_)
    w = fit(X[:half], r[:half])
    vh = X[half:] @ w
    rh = r[half:]
    wins = ties = tot = 0
    for i in range(len(rh)):
        for j in range(i + 1, len(rh)):
            if rh[i] == rh[j]:
                continue
            tot += 1
            better = i if rh[i] > rh[j] else j
            worse = j if better == i else i
            if vh[better] > vh[worse]:
                wins += 1
            elif vh[better] == vh[worse]:
                ties += 1
    acc = (wins + 0.5 * ties) / tot if tot else 0.0
    return fit(X, r).tolist(), round(acc, 4)


WORLDS = [
    ("resource1", FamilyConfig(collect=1, resource=1)),
    ("deceptive1", FamilyConfig(avoid=1, collect=1, deceptive=1)),
    ("resource1_avoid1", FamilyConfig(collect=1, resource=1, avoid=1)),
    ("resource1_deceptive1", FamilyConfig(collect=1, resource=1,
                                          avoid=1, deceptive=1)),
]

from experiments.games_amodal import fast_stack as _fs
FAST_ENC = _fs.ENC_CANDIDATES

APPROACH1 = (((0, 1), (2, 3), 1),)
APPROACH2 = (((0, 1), (4, 5), 1),)
AVOID2 = (((0, 1), (4, 5), -1),)
COMP_A1AV = (((0, 1), (2, 3), 1), ((0, 1), (4, 5), -1))
SCHED_RES = ("schedule", [APPROACH2, APPROACH1])
V1 = (((0, 1), (6, 7), 1),)
V2 = (((0, 1), (4, 5), 1),)          # under valued2x, 4/5 IS valued
SCHED_V2V1 = ("schedule", [V2, V1])  # the compositional form

BASE_LIB = [("plain", g) for g in
            (APPROACH1, APPROACH2, AVOID2, COMP_A1AV, SCHED_RES)]
VAL_LIB = [("valued2x", g) for g in (V1, SCHED_V2V1)]


def eval_goal(config, bank, goal, enc_fn):
    globals()["enc"] = enc_fn
    return play(config, "bank", bank, args.seed * 977, goal,
                plant_executor, args.episodes, args.steps)


def score_slow(config, bank, goal, enc_fn, episodes, seed):
    globals()["enc"] = enc_fn
    return play(config, "bank", bank, seed, goal, plant_executor,
                episodes, args.search_steps)


def race_library(config, lib, encs, banks):
    scored = []
    for kind, goal in lib:
        if kind not in encs:
            continue
        s = min(score_slow(config, banks[kind], goal, encs[kind], 24,
                           args.seed * 977 + off) for off in (1, 2))
        scored.append((s, kind, goal))
    scored.sort(key=lambda x: -x[0])
    best = None
    for s, kind, goal in scored[:3]:
        s2 = min(score_slow(config, banks[kind], goal, encs[kind], 48,
                            args.seed * 977 + off) for off in (3, 4))
        if best is None or s2 > best[0]:
            best = (s2, kind, goal)
    return best


results = {}
for name, config in WORLDS:
    w1, acc1 = fit_value_w_plane(config, args.seed * 131, 1)
    w2, acc2 = fit_value_w_plane(config, args.seed * 137, 2)
    row = {"rank_acc_p1": acc1, "rank_acc_p2": acc2}
    enc_std = ENC_CANDIDATES["second2"]
    globals()["enc"] = enc_std
    bank_std = build_bank(config, args.seed * 31, plant_executor)
    encs = {"plain": enc_std}
    banks = {"plain": bank_std}
    if w1 is not None and w2 is not None:
        v2x = make_valued2x_enc(w1, w2)
        globals()["enc"] = v2x
        banks["valued2x"] = build_bank(config, args.seed * 31,
                                       plant_executor)
        encs["valued2x"] = v2x
    for arm, lib in (("baseline", BASE_LIB),
                     ("valued2x", BASE_LIB + VAL_LIB)):
        best = race_library(config, lib, encs, banks)
        if best is None:
            continue
        s, kind, goal = best
        row[arm] = {"form": kind, "goal": str(goal),
                    "selection": round(s, 4),
                    "bank": eval_goal(config, banks[kind], goal,
                                      encs[kind])}
    globals()["enc"] = enc_std
    row["random"] = play(config, "random", None, args.seed * 977, None,
                         None, args.episodes, args.steps)
    results[name] = row
    cells = "  ".join(f"{a} {row[a]['bank']:+.3f}({row[a]['form'][:8]})"
                      for a in ("baseline", "valued2x") if a in row)
    print(f"  {name:<21} rnd {row['random']:+.3f}  {cells}  "
          f"(acc p1 {acc1} p2 {acc2})", flush=True)

report["results"] = results
print(json.dumps(report, indent=2))
if args.json:
    with open(args.json, "w") as handle:
        json.dump(report, handle, indent=2)

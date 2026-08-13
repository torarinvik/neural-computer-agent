"""ENTITY BINDING: persistent identity closes the value gap (F235).

F234 proved binding sufficient (privileged arm recovers 73% of the
bait tax) and identity-free value learning insufficient (ranking
0.842, the audit's L3 cap, loses selection). This probe adds the
witnessed perception primitive: PERSISTENT ENTITY TRACKING.

For this family items are static, so correspondence is exact and
generic: a plane-1 cell that stays occupied is the same entity;
appearance is a spawn (record the entity's relational context THEN);
disappearance is consumption. No object classes, no game names --
just cell-occupancy persistence and a per-entity memory of its
spawn-time context.

phi(o) grows from [1, d(o,av), d_manh(o,plane2), d_row(o,plane2)] to
include [d_manh_at_spawn, d_row_at_spawn]. Everything else is the
F234 harness unchanged: value-directed tracker in slots 6/7,
canonical REACH goals for the valued arms, arms none / true /
learned / shuffled.

Registered predictions:
  1. Learned-with-identity values rank >= 0.90 held-out (the audit's
     L4 level).
  2. The learned arm's deceptive1 effect climbs from F234's +0.054
     to >= +0.15 (toward the true arm's +0.222); tax from -0.36 to
     >= -0.15.
  3. Controls, resource1 (+0.586) and the shuffled control hold.
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

    def __init__(self):
        self.rows = None

    def reset(self, batch):
        self.rows = [dict() for _ in range(batch)]

    def update(self, frames):
        batch = frames.shape[0]
        if self.rows is None or len(self.rows) != batch:
            self.reset(batch)
        for i in range(batch):
            cells = {(int(r), int(c)) for r, c in
                     (frames[i, 1] > 0).nonzero()}
            hz = [(int(r), int(c)) for r, c in
                  (frames[i, 2] > 0).nonzero()]
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


TRACKER = EntityTable()


def _phi_cells(frames, ar, ac):
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
                best = max(row_cells,
                           key=lambda cp: sum(a * b for a, b in
                                              zip(w, cp[1])))
                out[i, 6], out[i, 7] = best[0]
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
    ("collect2", FamilyConfig(collect=2)),
    ("delayed3", FamilyConfig(delayed=3)),
    ("resource1", FamilyConfig(collect=1, resource=1)),
    ("avoid1_collect1", FamilyConfig(avoid=1, collect=1)),
    ("deceptive1", FamilyConfig(avoid=1, collect=1, deceptive=1)),
    ("deceptive2", FamilyConfig(avoid=1, collect=1, deceptive=2)),
]

from experiments.games_amodal import fast_stack as _fs
FAST_ENC = _fs.ENC_CANDIDATES

results = {}
for name, config in WORLDS:
    w_learned, rank_acc = fit_value_w(config, args.seed * 131)
    g_sh = torch.Generator().manual_seed(args.seed * 7 + 1)
    w_shuffled = None
    if w_learned is not None:
        perm = torch.randperm(len(w_learned), generator=g_sh).tolist()
        w_shuffled = [w_learned[p] for p in perm]
    arms = {"none": None}
    arms["true"] = make_valued_enc("true")
    if w_learned is not None:
        arms["learned"] = make_valued_enc("learned", w_learned)
        arms["shuffled"] = make_valued_enc("shuffled", w_shuffled)
    row = {"rank_acc": rank_acc}
    banks_std = {}
    for label, candidate in ENC_CANDIDATES.items():
        globals()["enc"] = candidate
        banks_std[label] = build_bank(config, args.seed * 31,
                                      plant_executor)
    for arm, valued_enc in arms.items():
        cands = dict(ENC_CANDIDATES)
        banks = dict(banks_std)
        if valued_enc is not None:
            cands = {"second2": ENC_CANDIDATES["second2"],
                     f"valued_{arm}": valued_enc}
            globals()["enc"] = valued_enc
            banks = {"second2": banks_std["second2"],
                     f"valued_{arm}": build_bank(config, args.seed * 31,
                                                 plant_executor)}
        best = None
        for label, candidate in cands.items():
            globals()["enc"] = candidate
            if label.startswith("valued"):
                # The valued tracker needs no race: its whole point is
                # that ONE canonical goal does the work. Score the
                # small canonical set on the slow verifier directly.
                canon = [(((0, 1), (6, 7), 1),),
                         (((0, 1), (6, 7), 1), ((0, 1), (4, 5), -1)),
                         (((0, 1), (2, 3), 1),)]
                goal, score = None, None
                for g0 in canon:
                    s = min(play(config, "bank", banks[label],
                                 args.seed * 977 + off, g0,
                                 plant_executor, args.search_episodes,
                                 args.search_steps) for off in (1, 2))
                    if score is None or s > score:
                        goal, score = g0, s
            else:
                picked = choose_by_race(config, banks[label], label)
                if picked is None:
                    continue
                goal, score = picked
            if best is None or score > best[0]:
                best = (score, label, goal)
        if best is None:
            continue
        sel_score, label, goal = best
        globals()["enc"] = (cands[label])
        bank = banks[label]
        if goal and goal[0] == "schedule":
            shown = ["schedule"] + [
                [[list(t[0]), list(t[1]), t[2]] for t in pg]
                for pg in goal[1]]
        elif goal and goal[0] == "guard":
            shown = ["guard"]
        else:
            shown = [[list(t[0]), list(t[1]), t[2]] for t in goal]
        row[arm] = {"encoder": label, "goal": shown,
                    "selection": round(sel_score, 4),
                    "bank": play(config, "bank", bank, args.seed * 977,
                                 goal, plant_executor, args.episodes,
                                 args.steps)}
    globals()["enc"] = ENC_CANDIDATES["second2"]
    row["random"] = play(config, "random", None, args.seed * 977, None,
                         None, args.episodes, args.steps)
    results[name] = row
    cells = "  ".join(f"{a} {row[a]['bank']:+.3f}" for a in
                      ("none", "true", "learned", "shuffled") if a in row)
    print(f"  {name:<16} rnd {row['random']:+.3f}  {cells}  "
          f"(rank_acc {rank_acc})", flush=True)

report["results"] = results
print(json.dumps(report, indent=2))
if args.json:
    with open(args.json, "w") as handle:
        json.dump(report, handle, indent=2)

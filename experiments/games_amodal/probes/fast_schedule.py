"""FULL SCHEDULE RACE: sequencing found by racing, not solo merit.

mechanism_schedule.py confirmed the schedule MECHANISM (chosen 6/6 on
resource1, zero regression elsewhere) but its gain was +0.065: the
phase pool came from the top-4 SOLO singles, and on resource1 both
clean legs -- approach resource, approach food -- score ~0 alone, so
greedy solo ranking never surfaces them. A sequencing basis cannot be
selected by solo merit; the schedule space must be raced directly.

The fast stack makes that affordable: this probe races EVERY ordered
pair of avatar-anchored approach singles (pa=(0,1), pb over ordered
pairs of usable non-avatar slots, sign +1; ~870 schedules) plus the
full plain-singles set, batched on the fast verifier with the trained
plant as executor, robust min-of-two-streams as always. EVALUATION
stays on the reference verifier so results pair with the archived
mb/ms runs. Worlds outside the fast verifier's support are skipped.

Registered predictions:
  1. resource1 reaches >= +0.8 (the clean resource->food schedule is
     in the raced set and beats every single by an order of
     magnitude).
  2. Controls stay at their baseline numbers: a plain single should
     still win everywhere else, and the schedule race must not
     displace it (F223 watch: bigger search space, so the
     selection-vs-evaluation gap is reported per world).
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json

import torch

from experiments.games_amodal.fast_family import FastFamilyVerifier
from experiments.games_amodal.fast_stack import (
    score_goals, score_schedules)
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


ENC_CANDIDATES = {"second2": make_enc("second"),
                  "approach2": make_enc("approach")}
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
    if goal is not None and goal and goal[0] == "schedule":
        schedule = goal[1]
    v = FamilyVerifier(config, batch_size=episodes, seed=seed)
    v.reset(seed=seed)
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
                if schedule is None:
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
        return FastFamilyVerifier(config, batch_size=batch,
                                  seed=args.seed * 31)

    fenc = FAST_ENC[label]

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
    if schedules:
        sched_scores = race(schedules, score_schedules)
        if float(sched_scores.max()) > best_score:
            j = int(sched_scores.argmax())
            best_goal = ("schedule", list(schedules[j]))
            best_score = float(sched_scores.max())
    return best_goal, best_score


WORLDS = [
    ("collect2", FamilyConfig(collect=2)),
    ("delayed3", FamilyConfig(delayed=3)),
    ("resource1", FamilyConfig(collect=1, resource=1)),
    ("avoid1_collect1", FamilyConfig(avoid=1, collect=1)),
    ("deceptive1", FamilyConfig(avoid=1, collect=1, deceptive=1)),
]

from experiments.games_amodal import fast_stack as _fs
FAST_ENC = _fs.ENC_CANDIDATES

results = {}
for name, config in WORLDS:
    banks = {}
    for label, candidate in ENC_CANDIDATES.items():
        globals()["enc"] = candidate
        banks[label] = build_bank(config, args.seed * 31, plant_executor)
    best = None
    for label in ENC_CANDIDATES:
        picked = choose_by_race(config, banks[label], label)
        if picked is None:
            continue
        goal, score = picked
        if best is None or score > best[0]:
            best = (score, label, goal)
    if best is None:
        continue
    sel_score, label, goal = best
    globals()["enc"] = ENC_CANDIDATES[label]
    bank = banks[label]
    if goal and goal[0] == "schedule":
        shown = ["schedule"] + [
            [[list(t[0]), list(t[1]), t[2]] for t in phase_goal]
            for phase_goal in goal[1]]
    else:
        shown = [[list(t[0]), list(t[1]), t[2]] for t in goal]
    row = {"encoder": label, "goal": shown,
           "selection": round(sel_score, 4),
           "bank": play(config, "bank", bank, args.seed * 977, goal,
                        plant_executor, args.episodes, args.steps),
           "random": play(config, "random", None, args.seed * 977, None,
                          None, args.episodes, args.steps)}
    results[name] = row
    kind = "SCHED" if goal and goal[0] == "schedule" else "single"
    print(f"  {name:<16} random {row['random']:+.3f}  bank "
          f"{row['bank']:+.3f}  [{kind}/{label}] {shown}", flush=True)

report["results"] = results
print(json.dumps(report, indent=2))
if args.json:
    with open(args.json, "w") as handle:
        json.dump(report, handle, indent=2)

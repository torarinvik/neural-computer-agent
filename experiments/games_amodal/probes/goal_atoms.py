"""GOAL ATOMS: the term shape was the bottleneck, not the slot count.

F226 v1/v2 (second_object.py) localized the width failure precisely.
Density (avoid3, avoid+collect) and urgency (intercept2) both refused
to make an 8-slot interface pay through PAIRED goal terms; the one
seed that escaped (1234, intercept2, +0.31) did it by abusing a pair
-- matching avatar ROW to the lowest faller's COLUMN -- because the
language cannot say the single-coordinate thing the world rewards:
"match your column to the column of the faller about to land".

This probe replaces the pair language with coordinate ATOMS. A term is
(state_slot, reference_slot, sign): cost += sign*|state[a] - ref[b]|.
Goals are built by greedy forward selection: score all admissible
atoms with the plant on the selection stream, then greedily add atoms
from the top 16 while the score improves, up to 4 terms. Atoms are a
strict expressivity superset of pairs (a pair is two atoms), with a
SMALLER single-term space (<=128 vs up to 1680).

Everything else is byte-compatible with second_object.py: same plant,
same banks, same encoder candidates (second/lowest x plane), same
worlds, same six/eight arms, same seeds -- so results pair per
(seed, world) against the archived so2 pair-language runs.

Registered predictions (v3):
  1. Controls (avoid1, collect2, intercept1) do not regress: atoms
     re-express approach/avoid at least as well as pairs.
  2. intercept2: atoms + the eight arm improve >= +0.2 over the
     archived pair composite, naming slot 7 (lowest-faller column)
     against slot 1 (avatar column) under encoder low1. A sign flip to
     positive is hoped for but UNCERTAIN.
  3. eight - six on intercept2 becomes consistently positive (width
     finally pays where the world demands it AND the language can
     spend it).
  4. F223 discipline: greedy 4-term selection has more freedom than
     pair selection, so selection-vs-evaluation deltas are reported
     per world; an advantage that lives only on the selection stream
     is selection bias and will be recorded as such.

v3 OUTCOME (6 seeds, recorded before v4 ran): intercept2 atoms-pairs
+0.30 (t=+1.60), catch policy [match col to faller col; maximize row
gap] convergent across seeds -- but avoid1/avoid3 REGRESSED (t=-2.31 /
-2.08): near-saturated worlds give ~zero selection signal to every
atom, and argmax over ties picks junk initial atoms like (0,0,+1).
Width settled: eight-six pooled t=+0.53 even with atoms.

v4 (both probes, for fairness): selection scores become min() over two
disjoint selection streams, and runs use --search-episodes 48
--search-steps 12 so rare-death worlds discriminate. Registered:
  5. avoid junk goals disappear; atoms match pairs on avoid worlds.
  6. intercept2's atom gain survives the pairs baseline getting the
     same selection upgrade (the wall is expressivity, not budget).
  7. Net atoms-pairs >= 0 pooled over all worlds.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json

import torch

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
parser.add_argument("--search-episodes", type=int, default=32)
parser.add_argument("--search-steps", type=int, default=10)
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


def _lowest(plane):
    """Generic extremal reduction: the set cell with the largest row
    index (scan-order sibling of F224's first/last), ABSENT if none.
    Ties broken by scan order via a stable flat argmax."""
    mask = plane.reshape(plane.shape[0], -1) > 0
    rank = (ROWS_IX * WIDTH + COLS_IX).reshape(-1).unsqueeze(0).float()
    scored = torch.where(mask, rank, torch.full_like(rank, -1.0))
    flat = scored.argmax(dim=1)
    present = mask.any(dim=1)
    row = torch.where(present, flat // WIDTH,
                      torch.full_like(flat, ABSENT))
    col = torch.where(present, flat % WIDTH,
                      torch.full_like(flat, ABSENT))
    return row, col


def make_enc(second_plane, kind="second"):
    def encoder(screen):
        frames = screen.view(-1, PLANES, HEIGHT, WIDTH)
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
            row, col = _kth_nearest(frames[:, second_plane],
                                    ar.clamp(max=VALUES-1),
                                    ac.clamp(max=VALUES-1), 1)
        else:
            row, col = _lowest(frames[:, second_plane])
        out[:, 6], out[:, 7] = row, col
        return out
    return encoder


ENC_CANDIDATES = {"second1": make_enc(1), "second2": make_enc(2),
                  "low1": make_enc(1, "low"), "low2": make_enc(2, "low")}
enc = ENC_CANDIDATES["second1"]           # rebound per world below


def goal_cost(state, reference, goal):
    total = None
    for a, b, sign in goal:
        term = sign * (state[:, a] - reference[:, b]).abs().float()
        total = term if total is None else total + term
    return total


def build_bank(config, seed, executor):
    probe = FamilyVerifier(config, batch_size=args.observations,
                           seed=seed + 7)
    probe.reset(seed=seed + 7)
    used = (enc(probe.observation()) < VALUES).float().mean(dim=0) >= 0.9
    bank = {}
    for act in range(4):
        v = FamilyVerifier(config, batch_size=args.observations,
                           seed=seed + act)
        v.reset(seed=seed + act)
        before = enc(v.observation())
        v.step(torch.full((args.observations,), act, dtype=torch.long))
        after = enc(v.observation())
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
    v = FamilyVerifier(config, batch_size=episodes, seed=seed)
    v.reset(seed=seed)
    g = torch.Generator().manual_seed(seed + 4242)
    total = torch.zeros(episodes)
    for _ in range(steps):
        if mode == "random":
            action = torch.randint(0, 4, (episodes,), generator=g)
        else:
            reference = enc(v.observation())
            reference = torch.where(reference < VALUES, reference,
                                    torch.zeros_like(reference))
            best, action = None, torch.zeros(episodes, dtype=torch.long)
            for act in range(4):
                if mode == "oracle":
                    shadow = copy.deepcopy(v)
                    shadow.step(torch.full((episodes,), act))
                    seen = enc(shadow.observation())
                    state = torch.where(seen < VALUES, seen,
                                        torch.zeros_like(seen))
                else:
                    program = bank.get(act)
                    state = (reference if program is None
                             else executor(program, reference))
                cost = goal_cost(state, reference, goal)
                if best is None:
                    best = cost.clone()
                else:
                    take = cost < best
                    best = torch.where(take, cost, best)
                    action = torch.where(
                        take, torch.full((episodes,), act), action)
        total += v.step(action).reward
    return float(total.mean())


def usable_slots(config):
    v = FamilyVerifier(config, batch_size=args.observations,
                       seed=args.seed * 31)
    v.reset(seed=args.seed * 31)
    code = enc(v.observation())
    present = (code < VALUES).float().mean(dim=0)
    return {s for s in range(SLOTS) if float(present[s]) >= 0.9}


pairs = [q for q in itertools.permutations(range(SLOTS), 2)]


def choose_atoms(config, bank, allowed):
    """Greedy forward selection over coordinate atoms, scored with the
    plant on the selection stream. Up to 4 terms; stop when no atom in
    the top-16 pool improves the score."""
    usable = usable_slots(config) & allowed
    atoms = [(a, b, sign) for a in usable for b in usable
             for sign in (1, -1)]
    if not atoms:
        return None

    def score(g):
        # v4: min over two disjoint selection streams. A goal that only
        # looks good on one stream is fitting that stream (F223), not
        # the world; min() makes such goals lose to robust ones.
        return min(play(config, "bank", bank, args.seed * 977 + off, g,
                        plant_executor, args.search_episodes,
                        args.search_steps) for off in (1, 2))

    scored = sorted(((score((t,)), t) for t in atoms), key=lambda x: -x[0])
    goal, best_r = [scored[0][1]], scored[0][0]
    pool = [t for _, t in scored[:16]]
    while len(goal) < 4:
        gain, pick = 0.0, None
        for t in pool:
            if t in goal:
                continue
            r = score(tuple(goal) + (t,))
            if r > best_r + gain:
                gain, pick = r - best_r, t
        if pick is None:
            break
        goal.append(pick)
        best_r += gain
    return tuple(goal)


WORLDS = [
    ("avoid1", FamilyConfig(avoid=1)),
    ("collect2", FamilyConfig(collect=2)),
    ("intercept1", FamilyConfig(intercept=1)),
    ("intercept2", FamilyConfig(intercept=2)),
    ("avoid2", FamilyConfig(avoid=2)),
    ("avoid3", FamilyConfig(avoid=3)),
    ("avoid2_collect1", FamilyConfig(avoid=2, collect=1)),
    ("avoid3_collect1", FamilyConfig(avoid=3, collect=1)),
]

ARMS = (("six", frozenset(range(6))), ("eight", frozenset(range(SLOTS))))

results = {}
for name, config in WORLDS:
    banks = {}
    for label, candidate in ENC_CANDIDATES.items():
        globals()["enc"] = candidate
        banks[label] = build_bank(config, args.seed * 31, plant_executor)
    row = {}
    for arm, allowed in ARMS:
        best = None
        for label, candidate in ENC_CANDIDATES.items():
            globals()["enc"] = candidate
            goal = choose_atoms(config, banks[label], allowed)
            if goal is None:
                continue
            score = min(play(config, "bank", banks[label],
                             args.seed * 977 + off, goal, plant_executor,
                             args.search_episodes, args.search_steps)
                        for off in (1, 2))
            if best is None or score > best[0]:
                best = (score, label, candidate, banks[label], goal)
        if best is None:
            continue
        sel_score, label, candidate, bank, goal = best
        globals()["enc"] = candidate
        row[arm] = {
            "encoder": label,
            "goal": [list(t) for t in goal],
            "selection": round(sel_score, 4),
            "bank": play(config, "bank", bank, args.seed * 977, goal,
                         plant_executor, args.episodes, args.steps),
            "oracle": play(config, "oracle", None, args.seed * 977, goal,
                           plant_executor, args.episodes, args.steps)}
    globals()["enc"] = ENC_CANDIDATES["second1"]
    row["random"] = play(config, "random", None, args.seed * 977, None,
                         None, args.episodes, args.steps)
    results[name] = row
    six = row.get("six", {}).get("bank")
    eight = row.get("eight", {}).get("bank")
    named = ("67" if eight is not None and any(
        t[0] in (6, 7) or t[1] in (6, 7)
        for t in row["eight"]["goal"]) else "--")
    print(f"  {name:<16} random {row['random']:+.3f}  six "
          f"{(six if six is not None else float('nan')):+.3f}  eight "
          f"{(eight if eight is not None else float('nan')):+.3f}  "
          f"[{named}]", flush=True)

report["results"] = results
print(json.dumps(report, indent=2))
if args.json:
    with open(args.json, "w") as handle:
        json.dump(report, handle, indent=2)

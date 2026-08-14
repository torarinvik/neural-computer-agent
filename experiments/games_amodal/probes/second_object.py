"""SECOND OBJECT: make width necessary, then test whether it pays.

F225 ended with a null that names this probe: at SLOTS=8 with composite
goals, slots 6/7 (the second-nearest tracker) were named in 0/24 cases.
The world family was never information-starved at six slots -- no world
makes the SECOND object strategically distinct. This probe builds such
worlds from existing family knobs only (component levels up to 3, and
component composition), then asks whether goal-language width finally
pays.

Worlds (all 8x8, no new verifier code):
  controls : avoid1 (slots 6/7 ABSENT -> arms must tie exactly),
             collect2 (width available but plausibly unneeded)
  anchor   : avoid2 (archived 6-slot and 8-slot numbers exist)
  dense    : avoid3, avoid2+collect1, avoid3+collect1 -- fleeing the
             nearest hazard can walk into the second one.

Causal contrast: ONE plant, ONE bank per (world, encoder); two goal
arms that differ only in which slots the goal search may name --
  six   : slots 0-5 (the F216/F222 goal space)
  eight : all 8 slots
Both arms run the F225 composite protocol (singles scored with the
plant on the selection stream, greedy pairing among the top 4). This is
slot_count.py's --composite path with the redundant truth-executor
pre-search removed (its result was discarded there; only the
admissibility gate is kept).

Registered predictions (v1, hazard density):
  1. eight - six > 0, paired per (seed, world), on the dense trio;
     expected t >= +2.
  2. The eight arm NAMES slots 6/7 on the dense trio in a majority of
     (seed, world) cells.
  3. avoid1: the arms are bit-identical (6/7 inadmissible), a built-in
     no-effect control. collect2: UNCERTAIN, recorded either way.
  4. The eight arm picks encoder second2 (second-nearest hazard) on the
     dense trio.
  5. F223 discipline: the eight arm's advantage must survive on the
     held evaluation stream (seed*977). Bigger search space winning on
     selection but not evaluation is selection bias, not width paying.

v1 OUTCOME (3 seeds, recorded before v2 ran): predictions 1, 2 and 4
REFUTED -- dense-trio paired delta t=+0.74, slots 6/7 named 2/9,
avoid3 arms bit-identical. Prediction 3 held exactly. Prediction 5's
tell appeared on avoid2_collect1 (selection +0.010, evaluation -0.010).
Hazard density does not make the second-NEAREST reduction necessary:
max-distance-to-nearest already sits within 0.09 of the zero ceiling.

v2: the flexible pair needs the RIGHT reduction, not just more copies
of `nearest`. Slots 6/7's reduction is now chosen per world (F224
planning selection) from four generic candidates: second-nearest of
plane 1/2, LOWEST set cell (extremal row, the scan-order sibling of
F224's first/last) of plane 1/2. Worlds add intercept1 (anchor) and
intercept2 -- the family's standing negative, where the urgent faller
(about to land, missing it is death) is often not the nearest one and
the six-slot language cannot say "under the lowest faller".

Registered predictions (v2):
  6. intercept2: the eight arm picks low1, names (0,1)~(6,7), and
     beats the six arm on evaluation -- the first width payoff.
  7. The avoid-trio null of v1 persists under the enlarged candidate
     set (no second-guessing a measured null).
  8. avoid1 still ties bit-identically; intercept1 ties or nearly ties
     (lowest == nearest when there is one faller; slots 6/7 duplicate
     2/3 there).
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


def choose_composite(config, bank, allowed):
    """F225's composite selection, restricted to `allowed` slots: score
    every admissible single term with the plant on the selection stream,
    then greedily try pairs among the top 4."""
    usable = usable_slots(config) & allowed
    singles = []
    for pa in pairs:
        for pb in pairs:
            if set(pa) & set(pb) or pa[0] > pa[1]:
                continue
            if not (set(pa) | set(pb)) <= usable:
                continue
            for sign in (1, -1):
                singles.append((pa, pb, sign))
    if not singles:
        return None

    def score(g):
        # v4 fairness upgrade, mirrored from goal_atoms.py: min over two
        # disjoint selection streams (F223 mitigation).
        return min(play(config, "bank", bank, args.seed * 977 + off, g,
                        plant_executor, args.search_episodes,
                        args.search_steps) for off in (1, 2))

    scored = sorted(((score((t,)), t) for t in singles), key=lambda x: -x[0])
    best_c, best_r = (scored[0][1],), scored[0][0]
    top = [t for _, t in scored[:4]]
    for i in range(len(top)):
        for j in range(len(top)):
            if i != j:
                r = score((top[i], top[j]))
                if r > best_r:
                    best_c, best_r = (top[i], top[j]), r
    return best_c


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
            goal = choose_composite(config, banks[label], allowed)
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
            "goal": [[list(t[0]), list(t[1]), t[2]] for t in goal],
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
        6 in t[0] + t[1] or 7 in t[0] + t[1]
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

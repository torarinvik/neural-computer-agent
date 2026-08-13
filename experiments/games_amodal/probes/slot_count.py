"""SLOT COUNT: the untested half of the amodality claim.

F220 scaled values and grid size with constants only, and said plainly
that SLOT COUNT was not tested. Every probe holds SLOTS=6: one object
tracked per plane. That width gates everything -- avoid2 plans against
its nearest hazard while the second one hits it, and richer world
families are unrepresentable.

This probe runs the full F216 protocol at SLOTS=8. The two new slots
are fed by one more generic reduction -- kth_nearest(channel, ref, k),
the obvious sibling of F224's nearest -- and WHICH object plane gets
the second tracker is chosen per world by planning selection, as in
F224. The plant retrains at the unchanged 40k budget; searches, goal
space and admissibility auto-scale from the SLOTS constant.

Registered predictions:
  1. The plant reaches its gate at the same 40k updates with 8 slots.
     UNCERTAIN -- this is the likeliest break point, and a miss is a
     scope boundary worth exactly as much as a pass.
  2. Multi-object worlds IMPROVE over their 6-slot numbers (avoid2,
     collect2, forage2): the second tracker is real information.
  3. Single-object worlds match their 6-slot numbers: the extra slots
     are ABSENT there and goal admissibility already ignores them.
  4. Search cost grows ~linearly (x8/6), not combinatorially.

The 6-slot reference is the archived gl2/co runs: same protocol, same
seeds, same worlds, so the comparison pairs per (seed, world).
"""

from __future__ import annotations

import argparse
import copy
import json

import torch

from experiments.games_amodal.game_family import (
    FamilyConfig, FamilyVerifier, family_variants)

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
parser.add_argument("--worlds", type=int, default=16)
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


def truth_executor(program, state):
    return run_parallel(state, program)


def _kth_nearest(plane, ref_row, ref_col, k):
    """Generic reduction: position of the k-th closest set cell to a
    reference coordinate pair. k=0 is F224's nearest."""
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


def make_enc(second_plane):
    """Slots 0-1: channel-0 peak. 2-3 / 4-5: nearest of planes 1 / 2.
    6-7: SECOND-nearest of `second_plane` -- which plane deserves the
    extra tracker is chosen per world by planning selection."""
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
        row, col = _kth_nearest(frames[:, second_plane],
                                ar.clamp(max=VALUES-1),
                                ac.clamp(max=VALUES-1), 1)
        out[:, 6], out[:, 7] = row, col
        return out
    return encoder


ENC_CANDIDATES = {"second1": make_enc(1), "second2": make_enc(2)}
enc = ENC_CANDIDATES["second1"]           # rebound per world below


def goal_cost(state, reference, goal):
    (a0, a1), (b0, b1), sign = goal
    reach = ((state[:, a0] - reference[:, b0]).abs()
             + (state[:, a1] - reference[:, b1]).abs()).float()
    return sign * reach


def build_bank(config, seed, executor):
    """F155/F192's row-versus-slot lesson, applied a third time.

    Requiring ALL slots present dropped EVERY row on the pure avoid
    worlds, where slots 2,3 are absent by construction -- the bank came
    back empty, the planner tied on every action, and both goal signs
    played constant action 0 with byte-identical returns. The sentinel
    for an unused SLOT masks the slot; only a missing avatar drops the
    row."""
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


def choose_goal(config, bank, signs):
    usable = usable_slots(config)
    best, best_reward = None, -1e9
    for a0 in range(SLOTS):
        for a1 in range(a0 + 1, SLOTS):
            for b0 in range(SLOTS):
                for b1 in range(SLOTS):
                    if b0 == b1 or {a0, a1} & {b0, b1}:
                        continue
                    if not {a0, a1, b0, b1} <= usable:
                        continue
                    for sign in signs:
                        reward = play(config, "bank", bank,
                                      args.seed * 977 + 1,
                                      ((a0, a1), (b0, b1), sign),
                                      truth_executor,
                                      args.search_episodes,
                                      args.search_steps)
                        if reward > best_reward:
                            best = ((a0, a1), (b0, b1), sign)
                            best_reward = reward
    return best


WORLDS = [
    ("collect1", FamilyConfig(collect=1)),
    ("collect2", FamilyConfig(collect=2)),
    ("intercept1", FamilyConfig(intercept=1)),
    ("intercept2", FamilyConfig(intercept=2)),
    ("avoid1", FamilyConfig(avoid=1)),
    ("avoid2", FamilyConfig(avoid=2)),
    ("forage2", FamilyConfig(forage=2)),
    ("forage2_inv", FamilyConfig(forage=2, inverted=True)),
]

results = {}
for name, config in WORLDS:
    best = None
    for label, candidate in ENC_CANDIDATES.items():
        globals()["enc"] = candidate
        bank = build_bank(config, args.seed * 31, plant_executor)
        goal = choose_goal(config, bank, (1, -1))
        if goal is None:
            continue
        score = play(config, "bank", bank, args.seed * 977 + 1, goal,
                     plant_executor, args.search_episodes,
                     args.search_steps)
        if best is None or score > best[0]:
            best = (score, label, candidate, bank, goal)
    if best is None:
        continue
    _, label, candidate, bank, goal = best
    globals()["enc"] = candidate
    row = {"encoder": label,
           "goal": [list(goal[0]), list(goal[1]), goal[2]],
           "random": play(config, "random", None, args.seed * 977, None,
                          None, args.episodes, args.steps),
           "bank": play(config, "bank", bank, args.seed * 977, goal,
                        plant_executor, args.episodes, args.steps),
           "oracle": play(config, "oracle", None, args.seed * 977, goal,
                          plant_executor, args.episodes, args.steps)}
    results[name] = row
    print(f"  {name:<13} random {row['random']:+.3f}  bank "
          f"{row['bank']:+.3f}  oracle {row['oracle']:+.3f}  "
          f"[{label}] goal {row['goal']}", flush=True)

report["results"] = results
for key in ("random", "bank", "oracle"):
    vals = [results[n][key] for n in results]
    report[f"mean_{key}"] = round(sum(vals) / len(vals), 4)

print(json.dumps(report, indent=2))
if args.json:
    with open(args.json, "w") as handle:
        json.dump(report, handle, indent=2)

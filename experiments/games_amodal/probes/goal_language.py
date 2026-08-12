"""Widen the goal LANGUAGE by one bit, and let control choose it.

Since F203, every arm including the oracle has been net NEGATIVE on the
avoid worlds. The recorded diagnosis was "the objective does not
describe intercept or avoid" — the system's only goal form is 'move
slot pair A toward slot pair B', and no assignment of slots makes
approaching the right answer in a world about staying away.

F214 and F215 changed what a goal is: not a thing a human writes but a
thing per-world control search chooses from a domain-general space. So
the fix for avoid is not a better hand-written objective, it is a WIDER
GOAL LANGUAGE for the same search. This probe adds one bit:

    goal = (pair_a, pair_b, sign)     sign in {+1 approach, -1 avoid}
    cost = sign * distance(pair_a -> pair_b)

Nothing about a sign is domain-specific — it is the smallest possible
extension of the existing form, and the search still names only slot
indices.

Registered predictions:
  1. With the signed language, the pure avoid worlds go POSITIVE under
     bank planning for the first time in the project's history, and the
     search picks sign=-1 there.
  2. The approach-only language stays negative on those worlds (this is
     F214's protocol re-run as the control arm).
  3. On collect worlds the search picks sign=+1 and matches F214 — the
     wider language costs nothing where the old one was right.
  4. The oracle ceiling with the signed language also goes positive on
     avoid worlds; if it does not, the language is still too narrow and
     the result reads as scope, not failure.

Selection integrity as in F214/F215: each world's goal is chosen on its
own cheap episodes (different seed stream), then frozen and evaluated on
fresh full-budget episodes with the frozen plant.
"""

from __future__ import annotations

import argparse
import copy
import json

import torch

from experiments.games_amodal.game_family import (
    FamilyVerifier, family_variants)

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

SLOTS, VALUES = 6, 8
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


def enc(screen):
    """The hand-written perception. Perception is orthogonal to the goal
    language and F215 measured its replacement separately; holding it
    fixed isolates this probe's one variable."""
    frames = screen.view(-1, PLANES, HEIGHT, WIDTH)
    out = torch.full((frames.shape[0], SLOTS), ABSENT, dtype=torch.long)
    for row in range(frames.shape[0]):
        avatar = frames[row, 0]
        if float(avatar.max()) <= 0:
            continue
        flat = int(avatar.reshape(-1).argmax())
        ar, ac = flat // WIDTH, flat % WIDTH
        out[row, 0], out[row, 1] = ar, ac
        for plane in (1, 2):
            base = 2 * plane
            mask = frames[row, plane] > 0
            if not bool(mask.any()):
                continue
            d = (ROWS_IX - ar).abs() + (COLS_IX - ac).abs()
            d = torch.where(mask, d, torch.full_like(d, 999))
            pick = int(d.reshape(-1).argmin())
            out[row, base], out[row, base + 1] = pick // WIDTH, pick % WIDTH
    return out


def goal_cost(state, reference, goal):
    (a0, a1), (b0, b1), sign = goal
    reach = ((state[:, a0] - reference[:, b0]).abs()
             + (state[:, a1] - reference[:, b1]).abs()).float()
    return sign * reach


def build_bank(config, seed, executor):
    bank = {}
    for act in range(4):
        v = FamilyVerifier(config, batch_size=args.observations,
                           seed=seed + act)
        v.reset(seed=seed + act)
        before = enc(v.observation())
        v.step(torch.full((args.observations,), act, dtype=torch.long))
        after = enc(v.observation())
        keep = (before < VALUES).all(dim=1) & (after < VALUES).all(dim=1)
        if int(keep.sum()) < 8:
            continue
        bank[act] = per_slot_search(before[keep][:args.examples],
                                    after[keep][:args.examples])
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


results = {}
for config in family_variants()[:args.worlds]:
    bank = build_bank(config, args.seed * 31, plant_executor)
    row = {"random": play(config, "random", None, args.seed * 977,
                          None, None, args.episodes, args.steps)}
    for label, signs in (("approach_only", (1,)), ("signed", (1, -1))):
        goal = choose_goal(config, bank, signs)
        if goal is None:
            row[label] = {"goal": None, "bank": row["random"],
                          "oracle": row["random"]}
            continue
        row[label] = {
            "goal": [list(goal[0]), list(goal[1]), goal[2]],
            "bank": play(config, "bank", bank, args.seed * 977, goal,
                         plant_executor, args.episodes, args.steps),
            "oracle": play(config, "oracle", None, args.seed * 977, goal,
                           plant_executor, args.episodes, args.steps)}
    results[config.name] = row
    summary = {k: (v if not isinstance(v, dict) else v["bank"])
               for k, v in row.items()}
    sign = (row["signed"]["goal"][2]
            if row["signed"]["goal"] is not None else "-")
    print(f"  {config.name:<24} random {row['random']:+.3f}  "
          f"approach {summary['approach_only']:+.3f}  "
          f"signed {summary['signed']:+.3f}  (sign {sign})", flush=True)

report["results"] = results
for label in ("approach_only", "signed"):
    report[f"mean_bank_{label}"] = round(
        sum(r[label]["bank"] for r in results.values()) / len(results), 4)
report["mean_random"] = round(
    sum(r["random"] for r in results.values()) / len(results), 4)

print(json.dumps(report, indent=2))
if args.json:
    with open(args.json, "w") as handle:
        json.dump(report, handle, indent=2)

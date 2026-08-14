"""CROSS-SUBSTRATE TRANSFER: grid experience warm-starts graphs (F242).

F241 proved the frozen stack plans on a geometry-free substrate. This
probe asks the founding question ACROSS substrates: does what the
grid taught make graph worlds cheaper?

Mechanism: ROLE ABSTRACTION. Promoted winners are slot-concrete, so
each substrate's perception declares a role table -- which slot
pattern implements REACH(entity-type) / AVOID(entity-type) in its
dictionary. Promotion abstracts the archived F240 grid winners into
roles mechanically (subset match on terms; schedules to SEQ of leg
roles; no curation, unmatched forms drop). The graph substrate
re-grounds the abstract forms in its own slots, and warm acquisition
races only those.

  grid roles   REACH1=((0,1),(2,3),+1)  REACH2=((0,1),(4,5),+1)
               AVOID2=((0,1),(4,5),-1)
  graph roles  REACH1=((0,2),(2,3),+1)  AVOID2=((1,2),(2,3),-1)
               (slot 0/1 = BFS distance to food/hazard; 2,3 origin)

Registered predictions:
  1. Warm (re-grounded roles, ~3 candidates) >= 90% of F241's
     cold-raced returns at <= 10% of the race cost.
  2. gcollect2 is FIXED by transfer: the role REACH beats F241's
     wandering race pick there (warm >= cold on gcollect2).
  3. Abstraction is mechanical: the archived F240 libraries match
     roles at a reported rate with zero human choices.
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
parser.add_argument("--search-episodes", type=int, default=48)
parser.add_argument("--search-steps", type=int, default=12)
parser.add_argument("--pd", default="", help="lp-<seed>.json with the promoted grid library")
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




from experiments.games_amodal.graph_world import (
    GraphConfig, GraphVerifier, NODES, PORTS)
from experiments.games_amodal.graph_perception import bfs_all


def encode(obs, edges):
    B = obs.shape[0]
    out = torch.full((B, SLOTS), ABSENT, dtype=torch.long)
    out[:, 2] = 0
    out[:, 3] = 0
    for i in range(B):
        if float(obs[i, 0].max()) <= 0:
            continue
        agent = int(obs[i, 0].argmax())
        dist = bfs_all(edges[i])
        foods = [int(n) for n in (obs[i, 1] > 0).nonzero().flatten()]
        if foods:
            foods.sort(key=lambda n: int(dist[agent, n]))
            n0 = foods[0]
            d0 = int(dist[agent, n0])
            out[i, 0] = min(d0, VALUES - 1)
            for p in range(PORTS):
                nxt = int(edges[i, agent, p])
                on_path = (d0 > 0
                           and int(dist[nxt, n0]) == d0 - 1)
                out[i, 4 + p] = 1 if on_path else 0
        hz = (obs[i, 2] > 0).nonzero().flatten()
        if hz.numel():
            out[i, 1] = min(int(dist[agent, int(hz[0])]), VALUES - 1)
    return out


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
    v = GraphVerifier(config, batch_size=args.observations,
                      seed=seed + 7)
    v.reset(seed=seed + 7)
    used = (encode(v.observation(), v.structure())
            < VALUES).float().mean(dim=0) >= 0.9
    bank, agreement = {}, {}
    for act in range(PORTS):
        v = GraphVerifier(config, batch_size=args.observations,
                          seed=seed + act)
        v.reset(seed=seed + act)
        before = encode(v.observation(), v.structure())
        v.step(torch.full((args.observations,), act, dtype=torch.long))
        after = encode(v.observation(), v.structure())
        keep = ((before[:, used] < VALUES).all(dim=1)
                & (after[:, used] < VALUES).all(dim=1))
        if int(keep.sum()) < 8:
            continue
        b = torch.where(before[keep] < VALUES, before[keep],
                        torch.zeros_like(before[keep]))
        a = torch.where(after[keep] < VALUES, after[keep],
                        torch.zeros_like(after[keep]))
        program = per_slot_search(b[:args.examples], a[:args.examples])
        bank[act] = program
        pred = run_parallel(b, program)  # program fit, truth-executed
        agreement[act] = round(float((pred[:, 0] == a[:, 0]).float()
                                     .mean()), 4)
    return bank, agreement


def play(config, mode, bank, seed, goal, executor, episodes, steps):
    v = GraphVerifier(config, batch_size=episodes, seed=seed)
    v.reset(seed=seed)
    g = torch.Generator().manual_seed(seed + 4242)
    total = torch.zeros(episodes)
    for _ in range(steps):
        if mode == "random":
            action = torch.randint(0, PORTS, (episodes,), generator=g)
        else:
            reference = encode(v.observation(), v.structure())
            reference = torch.where(reference < VALUES, reference,
                                    torch.zeros_like(reference))
            best, action = None, torch.zeros(episodes, dtype=torch.long)
            for act in range(PORTS):
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
    v = GraphVerifier(config, batch_size=args.observations,
                      seed=args.seed * 31)
    v.reset(seed=args.seed * 31)
    code = encode(v.observation(), v.structure())
    present = (code < VALUES).float().mean(dim=0)
    return {s for s in range(SLOTS) if float(present[s]) >= 0.9}


import itertools
pairs = [q for q in itertools.permutations(range(SLOTS), 2)]


GRID_ROLES = {
    (((0, 1), (2, 3)), 1): "REACH1",
    (((0, 1), (4, 5)), 1): "REACH2",
    (((0, 1), (4, 5)), -1): "AVOID2",
    (((0, 1), (6, 7)), 1): "REACH_VALUED",
}

GRAPH_FORMS = {
    "REACH1": (((0, 2), (2, 3), 1),),
    "AVOID2": (((1, 2), (2, 3), -1),),
}


def abstract_library(lib_entries):
    """Mechanically abstract archived grid winners into roles.

    A term matches a role by its (a-pair, b-pair, sign) shape; a
    composite contributes every matching role; a schedule becomes the
    SEQ of its legs' roles (dropped if any leg is roleless). Unmatched
    forms drop. Zero curation."""
    roles = []
    for kind, goal_str in lib_entries:
        goal = eval(goal_str)  # archived by our own probe
        if goal and goal[0] == "schedule":
            legs = []
            for leg in goal[1]:
                m = [GRID_ROLES.get(((tuple(t[0]), tuple(t[1])), t[2]))
                     for t in leg]
                m = [x for x in m if x]
                legs.append(m[0] if m else None)
            if all(legs):
                roles.append(("SEQ", tuple(legs)))
        else:
            for t in goal:
                role = GRID_ROLES.get(((tuple(t[0]), tuple(t[1])),
                                       t[2]))
                if role:
                    roles.append(("ROLE", role))
    return roles


def reground(roles):
    """Re-ground abstract roles in the graph slot dictionary."""
    out, seen = [], set()

    def add(goal):
        key = str(goal)
        if key not in seen:
            seen.add(key)
            out.append(goal)

    have = {r for k, r in roles if k == "ROLE"}
    for k, payload in roles:
        if k == "ROLE" and payload in GRAPH_FORMS:
            add(GRAPH_FORMS[payload])
    if {"REACH1", "AVOID2"} <= (have | {r for k, rs in roles
                                        if k == "SEQ" for r in rs
                                        if r in GRAPH_FORMS}):
        add(GRAPH_FORMS["REACH1"] + GRAPH_FORMS["AVOID2"])
    return out


WORLDS = [
    ("gcollect1", GraphConfig(collect=1)),
    ("gcollect2", GraphConfig(collect=2)),
    ("gcollect1_avoid1", GraphConfig(collect=1, avoid=1)),
]

import json as _json
archived = _json.load(open(args.pd))
lib_entries = archived["results"]["library"]
roles = abstract_library(lib_entries)
matched = len(roles)
candidates = reground(roles)
print(f"  abstracted {matched} roles from {len(lib_entries)} grid "
      f"winners -> {len(candidates)} graph candidates: "
      f"{[str(c) for c in candidates]}", flush=True)

results = {"roles": [str(r) for r in roles],
           "candidates": [str(c) for c in candidates],
           "match_rate": round(matched / max(1, len(lib_entries)), 3)}
for name, config in WORLDS:
    bank, agreement = build_bank(config, args.seed * 31, plant_executor)
    best, best_score, spent = None, None, 0
    for goal in candidates:
        s = min(play(config, "bank", bank, args.seed * 977 + off, goal,
                     plant_executor, args.search_episodes,
                     args.search_steps) for off in (1, 2))
        spent += 2 * args.search_episodes
        if best_score is None or s > best_score:
            best, best_score = goal, s
    if best is None:
        continue
    row = {"goal": str(best), "episodes_spent": spent,
           "random": play(config, "random", None, args.seed * 977,
                          None, None, args.episodes, args.steps),
           "bank": play(config, "bank", bank, args.seed * 977, best,
                        plant_executor, args.episodes, args.steps)}
    results[name] = row
    print(f"  {name:<18} random {row['random']:+.3f}  warm "
          f"{row['bank']:+.3f} ({spent}ep)  goal {row['goal']}",
          flush=True)

report["results"] = results
print(_json.dumps(report, indent=2))
if args.json:
    with open(args.json, "w") as handle:
        _json.dump(report, handle, indent=2)

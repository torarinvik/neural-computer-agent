"""Plan with a bank the READER wrote, and see whether the search is
still needed for anything.

F203 closed the loop with a SEARCHED bank: recipes found by enumerating
candidates against the frozen interpreter, which matched the true
simulator on three of four games. The search was still in the loop, so
every new world cost an enumeration and nothing was learned from having
solved the last thousand.

This replaces that one component and changes nothing else. Four arms:

  random    acting without a model.
  SEARCHED  F203's bank, rebuilt in the parallel language: per-slot
            enumeration against the frozen plant.
  READ      the same bank, produced by ONE forward pass of a reader that
            was trained on the search's own labels and never saw this
            world.
  oracle    planning with the real environment, by copying the verifier
            and stepping it. The ceiling, and the thing that made F203's
            three objective bugs visible instead of readable as model
            error.

If READ tracks SEARCHED, the enumeration is no longer load-bearing at
test time: a new world costs a forward pass.

The goal is SUPPLIED, not learned, exactly as in F203, and the planner
is greedy at depth one for the reason recorded there.
"""

from __future__ import annotations

import argparse
import copy
import json

import torch

from experiments.games_amodal.game_family import (
    FamilyVerifier, family_variants)
from experiments.games_amodal.probes.schema_families import (
    RandomFamily, random_family_spec)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=69316)
parser.add_argument("--dim", type=int, default=128)
parser.add_argument("--interpreter-updates", type=int, default=40000)
parser.add_argument("--reader-updates", type=int, default=8000)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight-decay", type=float, default=0.01)
parser.add_argument("--examples", type=int, default=32)
parser.add_argument("--pool", type=int, default=4000)
parser.add_argument("--observations", type=int, default=256)
parser.add_argument("--episodes", type=int, default=64)
parser.add_argument("--steps", type=int, default=12)
parser.add_argument("--train-source", default="families",
                    choices=("families", "primitives", "mixed"))
parser.add_argument("--synthetic-share", type=float, default=0.0,
                    help="fraction of the wake pool drawn as random "
                         "parallel programs over random states. See "
                         "factored.py: 0.3 is the measured optimum.")
parser.add_argument("--wake-games", type=int, default=7,
                    help="grid variants the wake phase may use; the rest "
                         "are held out. Must not overlap --game-from.")
parser.add_argument("--game-from", type=int, default=0)
parser.add_argument("--game-to", type=int, default=4)
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

    def code_for(self, s, program):
        op, j, m = program[s]
        return (self.slot(torch.tensor(s)) + self.op(torch.tensor(op))
                + self.arg_j(torch.tensor(j)) + self.arg_m(torch.tensor(m)))

    def forward(self, program, state):
        """One residual step per slot, with the PRE-state re-supplied at
        every one. See `factored.py` for the sweep that decided this: a
        summed program code reaches 0.4055, this reaches 1.0000."""
        hot = torch.nn.functional.one_hot(
            state, VALUES).float().view(state.shape[0], -1)
        base = self.load(hot)
        latent = base
        for s in range(SLOTS):
            code = self.code_for(s, program).unsqueeze(0).expand(
                latent.shape[0], -1)
            latent = self.norm(latent + self.step(
                torch.cat([latent, base, code], dim=-1)))
        return self.head(latent).view(-1, SLOTS, VALUES)


def random_parallel(generator):
    out = []
    for s in range(SLOTS):
        op = int(torch.randint(0, len(PAR_OPS), (1,), generator=generator))
        j = int(torch.randint(0, SLOTS, (1,), generator=generator))
        if j == s:
            j = (j + 1) % SLOTS
        m = int(torch.randint(0, len(MODULI), (1,), generator=generator))
        out.append((op, j, m))
    return out


interp = Interpreter(args.dim)
opt = torch.optim.AdamW(interp.parameters(), lr=args.lr,
                        weight_decay=args.weight_decay)
train_gen = torch.Generator().manual_seed(args.seed * 104729)
for _ in range(args.interpreter_updates):
    prog = random_parallel(train_gen)
    st = torch.randint(0, VALUES, (args.batch_size, SLOTS),
                       generator=train_gen)
    tgt = run_parallel(st, prog)
    loss = torch.nn.functional.cross_entropy(
        interp(prog, st).reshape(-1, VALUES), tgt.reshape(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
for parameter in interp.parameters():
    parameter.requires_grad_(False)

check_gen = torch.Generator().manual_seed(args.seed + 5551)
hits = rows = 0
for _ in range(32):
    prog = random_parallel(check_gen)
    st = torch.randint(0, VALUES, (128, SLOTS), generator=check_gen)
    tgt = run_parallel(st, prog)
    with torch.no_grad():
        hits += int((interp(prog, st).argmax(-1) == tgt).sum())
    rows += tgt.numel()
report = {"seed": args.seed, "train_source": args.train_source,
          "interpreter_check": round(hits / rows, 4), "steps": args.steps}


def slot_state(screen):
    frames = screen.view(-1, PLANES, HEIGHT, WIDTH)
    batch = frames.shape[0]
    out = torch.full((batch, SLOTS), ABSENT, dtype=torch.long)
    ri = torch.arange(HEIGHT).view(-1, 1).expand(HEIGHT, WIDTH)
    ci = torch.arange(WIDTH).view(1, -1).expand(HEIGHT, WIDTH)
    for row in range(batch):
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
            d = (ri - ar).abs() + (ci - ac).abs()
            d = torch.where(mask, d, torch.full_like(d, 999))
            pick = int(d.reshape(-1).argmin())
            out[row, base], out[row, base + 1] = pick // WIDTH, pick % WIDTH
    return out


def sanitise(before, after):
    alive = (before[:, 0] < VALUES) & (after[:, 0] < VALUES)
    before, after = before[alive], after[alive]
    if before.shape[0] == 0:
        return None
    used = (before < VALUES).all(dim=0) & (after < VALUES).all(dim=0)
    return (torch.where(before < VALUES, before, torch.zeros_like(before)),
            torch.where(after < VALUES, after, torch.zeros_like(after)),
            used)


def per_slot_search(before, after):
    program, cost = [], 0
    for s in range(SLOTS):
        want = after[:, s]
        best, best_score = NOOP, -1.0
        for op in range(len(PAR_OPS)):
            for j in range(SLOTS):
                if j == s and PAR_OPS[op] in ("CINC", "CDEC", "COPY"):
                    continue
                for m in range(len(MODULI)):
                    cost += 1
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
    return program, cost


# ------------------------------------------------------------ the reader
class Reader(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.embed = torch.nn.Sequential(
            torch.nn.Linear(2 * SLOTS * VALUES, dim), torch.nn.ReLU(),
            torch.nn.Linear(dim, dim), torch.nn.ReLU())
        self.pool = torch.nn.Sequential(
            torch.nn.Linear(dim, dim), torch.nn.ReLU(),
            torch.nn.Linear(dim, dim), torch.nn.ReLU())
        self.op = torch.nn.Linear(dim, SLOTS * len(PAR_OPS))
        self.arg_j = torch.nn.Linear(dim, SLOTS * SLOTS)
        self.arg_m = torch.nn.Linear(dim, SLOTS * len(MODULI))

    def forward(self, before, after):
        b, e, _ = before.shape
        hot = torch.cat([
            torch.nn.functional.one_hot(before, VALUES).float().view(b, e, -1),
            torch.nn.functional.one_hot(after, VALUES).float().view(b, e, -1)],
            dim=-1)
        latent = self.pool(self.embed(hot).mean(dim=1))
        return (self.op(latent).view(b, SLOTS, len(PAR_OPS)),
                self.arg_j(latent).view(b, SLOTS, SLOTS),
                self.arg_m(latent).view(b, SLOTS, len(MODULI)))


def game_transitions(config, count, seed, action):
    verifier = FamilyVerifier(config, batch_size=count, seed=seed)
    verifier.reset(seed=seed)
    before = slot_state(verifier.observation())
    verifier.step(torch.full((count,), action, dtype=torch.long))
    return before, slot_state(verifier.observation())


def family_transitions(family, count, action, generator):
    size = len(family.states)
    idx = torch.randint(0, size, (count,), generator=generator)
    nxt = torch.tensor([family.table[int(x)][action] for x in idx])
    return family.slot_values(idx), family.slot_values(nxt)


def wake(count, generator, source):
    """Worlds the reader learns from. The games under test are NEVER in
    here: `families` uses no grid world at all, `primitives` uses the
    seven single-mechanic games and is only paired with a compound test
    slice."""
    primitives = family_variants()[:args.wake_games]
    rows, guard = [], 0
    while len(rows) < count:
        guard += 1
        if guard > count * 50:
            raise SystemExit("wake phase could not fill the pool")
        if (args.synthetic_share > 0
                and float(torch.rand(1, generator=generator))
                < args.synthetic_share):
            prog = random_parallel(generator)
            before = torch.randint(0, VALUES, (args.examples, SLOTS),
                                   generator=generator)
            after = run_parallel(before, prog)
            rows.append((before, after, per_slot_search(before, after)[0]))
            continue
        take_game = (source == "primitives"
                     or (source == "mixed" and len(rows) % 2 == 0))
        draw = 2 * args.examples      # episodes END; sanitise drops rows
        if take_game:
            config = primitives[int(torch.randint(
                0, len(primitives), (1,), generator=generator))]
            seed = int(torch.randint(0, 10 ** 6, (1,), generator=generator))
            action = int(torch.randint(0, 4, (1,), generator=generator))
            pair = game_transitions(config, draw, seed, action)
        else:
            family = RandomFamily(random_family_spec(generator))
            action = int(torch.randint(0, family.actions, (1,),
                                       generator=generator))
            pair = family_transitions(family, draw, action, generator)
        clean = sanitise(*pair)
        if clean is None or clean[0].shape[0] < args.examples:
            continue
        before = clean[0][:args.examples]
        after = clean[1][:args.examples]
        program, _ = per_slot_search(before, after)
        rows.append((before, after, program))
    return (torch.stack([r[0] for r in rows]),
            torch.stack([r[1] for r in rows]),
            torch.tensor([[list(t) for t in r[2]] for r in rows]))


pool = wake(args.pool, torch.Generator().manual_seed(args.seed * 15485863),
            args.train_source)
reader = Reader(args.dim)
r_opt = torch.optim.AdamW(reader.parameters(), lr=args.lr,
                          weight_decay=args.weight_decay)
gen = torch.Generator().manual_seed(args.seed * 104729)
for _ in range(args.reader_updates):
    pick = torch.randint(0, pool[0].shape[0], (args.batch_size,),
                         generator=gen)
    before, after, labels = (t[pick] for t in pool)
    po, pj, pm = reader(before, after)
    loss = (torch.nn.functional.cross_entropy(
                po.reshape(-1, len(PAR_OPS)), labels[:, :, 0].reshape(-1))
            + torch.nn.functional.cross_entropy(
                pj.reshape(-1, SLOTS), labels[:, :, 1].reshape(-1))
            + torch.nn.functional.cross_entropy(
                pm.reshape(-1, len(MODULI)), labels[:, :, 2].reshape(-1)))
    r_opt.zero_grad()
    loss.backward()
    r_opt.step()
for parameter in reader.parameters():
    parameter.requires_grad_(False)


def goal_cost(state, reference=None):
    """F203's objective, unchanged. Score the NEW avatar position against
    the OLD object position, so reaching the target does not move it."""
    target = state if reference is None else reference
    reach = (state[:, 0] - target[:, 2]).abs() \
        + (state[:, 1] - target[:, 3]).abs()
    absent = (target[:, 2] >= VALUES) | (target[:, 3] >= VALUES)
    return torch.where(absent, torch.full_like(reach, 99), reach).float()


def build_bank(config, seed, how: str, evidence_from=None):
    """One recipe per action. `searched` enumerates; `read` does not look
    at a single candidate.

    `evidence_from` builds the bank from ANOTHER world's transitions
    while still planning in this one. That is the control the end-to-end
    result needs most: every grid world here shares its avatar dynamics,
    so a bank read from the wrong world is still right about how the
    avatar moves. If planning with it works as well, the reader is
    supplying shared action semantics rather than reading THIS world, and
    the honest claim shrinks accordingly."""
    source = evidence_from if evidence_from is not None else config
    bank, cost = {}, 0
    for act in range(4):
        before, after = game_transitions(source, args.observations,
                                         seed + act, act)
        clean = sanitise(before, after)
        if clean is None or clean[0].shape[0] < 8:
            continue
        b, a, _ = clean
        if how == "searched":
            program, spent = per_slot_search(b[:args.examples],
                                             a[:args.examples])
            cost += spent
        else:
            with torch.no_grad():
                po, pj, pm = reader(b[:args.examples].unsqueeze(0),
                                    a[:args.examples].unsqueeze(0))
            program = [(int(po[0, s].argmax()), int(pj[0, s].argmax()),
                        int(pm[0, s].argmax())) for s in range(SLOTS)]
            cost += 1
        bank[act] = program
    return bank, cost


def play(config, mode, bank, seed):
    verifier = FamilyVerifier(config, batch_size=args.episodes, seed=seed)
    verifier.reset(seed=seed)
    generator = torch.Generator().manual_seed(seed + 4242)
    total = torch.zeros(args.episodes)
    for _ in range(args.steps):
        if mode == "random":
            action = torch.randint(0, verifier.action_count,
                                   (args.episodes,), generator=generator)
        else:
            best_cost = None
            reference = slot_state(verifier.observation())
            action = torch.zeros(args.episodes, dtype=torch.long)
            for act in range(verifier.action_count):
                if mode == "oracle":
                    shadow = copy.deepcopy(verifier)
                    shadow.step(torch.full((args.episodes,), act))
                    cost = goal_cost(slot_state(shadow.observation()),
                                     reference)
                else:
                    state = slot_state(verifier.observation())
                    state = torch.where(state < VALUES, state,
                                        torch.zeros_like(state))
                    program = bank.get(act)
                    if program is not None:
                        with torch.no_grad():
                            state = interp(program, state).argmax(-1)
                    cost = goal_cost(state, reference)
                if best_cost is None:
                    best_cost = cost.clone()
                else:
                    take = cost < best_cost
                    best_cost = torch.where(take, cost, best_cost)
                    action = torch.where(
                        take, torch.full((args.episodes,), act), action)
        total += verifier.step(action).reward
    return round(float(total.mean()), 4)


MODE_PROGRAM = None
_counts: dict = {}
for _row in range(pool[2].shape[0]):
    _key = tuple(tuple(int(v) for v in pool[2][_row, _s]) for _s in range(SLOTS))
    _counts[_key] = _counts.get(_key, 0) + 1
MODE_PROGRAM = [tuple(t) for t in max(_counts.items(),
                                      key=lambda kv: kv[1])[0]]
report["mode_program"] = [list(t) for t in MODE_PROGRAM]

tested = family_variants()[args.game_from:args.game_to]
results = {}
for index, config in enumerate(tested):
    other = tested[(index + 1) % len(tested)]
    searched, s_cost = build_bank(config, args.seed * 31, "searched")
    read, r_cost = build_bank(config, args.seed * 31, "read")
    stray, _ = build_bank(config, args.seed * 31, "read", evidence_from=other)
    mode_bank = {act: MODE_PROGRAM for act in range(4)}
    agree = 0
    for act in searched:
        if act in read:
            agree += sum(int(searched[act][s] == read[act][s])
                         for s in range(SLOTS))
    results[config.name] = {
        "random": play(config, "random", None, args.seed * 977),
        "mode": play(config, "bank", mode_bank, args.seed * 977),
        "read_wrong_world": play(config, "bank", stray, args.seed * 977),
        "searched": play(config, "bank", searched, args.seed * 977),
        "read": play(config, "bank", read, args.seed * 977),
        "oracle": play(config, "oracle", None, args.seed * 977),
        "wrong_world_source": str(getattr(other, "name", other)),
        "searched_candidates": s_cost, "read_forward_passes": r_cost,
        "slot_agreement": round(agree / max(6 * len(searched), 1), 4)}
report["results"] = results
for arm in ("random", "mode", "read_wrong_world", "searched", "read",
            "oracle"):
    report[f"mean_{arm}"] = round(
        sum(v[arm] for v in results.values()) / max(len(results), 1), 4)
report["mean_slot_agreement"] = round(
    sum(v["slot_agreement"] for v in results.values())
    / max(len(results), 1), 4)

print(json.dumps(report, indent=2))
if args.json:
    with open(args.json, "w") as handle:
        json.dump(report, handle, indent=2)

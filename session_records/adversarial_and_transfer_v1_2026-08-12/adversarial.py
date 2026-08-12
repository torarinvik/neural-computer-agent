"""Attacks on F204-F208, built to make them fail.

The controls in F204-F208 were the ordinary kind: floors, shuffled
labels, held-out splits, a wrong-world arm. Those establish that a
result is not nothing. They do not establish that it survives an
opponent, because none of them was CHOSEN to break it. Everything here
was.

Each attack states what it would take to falsify the claim, and a
prediction made before the run. The predictions are in the code so they
cannot be revised after seeing the numbers.

ATTACK 1 -- ADVERSARIAL WORLDS FOR THE LANGUAGE.
    F204 claims parallel recipes match or beat sequential depth-2. That
    claim was measured on worlds I did not choose. Parallel semantics
    provably cannot express `INC 0; INC 0`, nor any chain where the
    second write READS what the first wrote. So build worlds made only
    of those.
    PREDICTION: parallel loses badly here, and F204's claim is exposed
    as a fact about the grid/rule distribution rather than about the
    languages. If parallel wins even here, my own statement that the
    expressivity loss is real was wrong.

ATTACK 2 -- HOSTILE BASELINES FOR THE READER.
    F205's floors were weak on purpose-built grounds: a single global
    mode program, and the identity. The strong competitor for grid
    worlds is "look up what action k usually does", because every world
    here shares its avatar dynamics. And the strong competitor for a
    learned reader is RETRIEVAL -- nearest neighbour in the wake pool,
    no learning at all.
    PREDICTION: per-action mode is close to the reader on grid worlds
    and far below on rule families. If the reader does not clear
    per-action mode on grids, F205 and F207 are substantially weakened.

ATTACK 3 -- SLOT PERMUTATION.
    The whole architecture rests on the interface being AMODAL: slots
    are addresses, not named quantities. Permuting the six slots
    produces a legitimate world the reader has never seen. The per-slot
    search is permutation-equivariant by construction and must be
    unaffected, so this isolates the reader.
    PREDICTION: the reader degrades badly. It has slot-specific heads
    and trained where slots 0 and 1 are always the avatar. If it
    collapses to the floor, the reader is NOT amodal even though the
    plant and the language are.

ATTACK 4 -- STARVE THE EVIDENCE.
    Both arms got 32 transitions. Fewer should hurt the search first
    (it fits whatever it is given) or the reader first (it may need the
    distribution to be visible).
    PREDICTION: the search degrades faster, because at 4 rows almost
    any instruction fits exactly and it takes the first one.

ATTACK 5 -- ADVERSARIAL PROGRAMS FOR THE PLANT.
    F204's plant gate uses RANDOM programs. Choose instead the programs
    that maximally couple slots -- every write a COPY or a conditional
    reading a different slot, with no NOOPs to make it easy.
    PREDICTION: the plant falls below its 0.9973 random-program figure.
"""

from __future__ import annotations

import argparse
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
parser.add_argument("--eval-rows", type=int, default=128)
parser.add_argument("--pool", type=int, default=4000)
parser.add_argument("--wake-games", type=int, default=15)
parser.add_argument("--synthetic-share", type=float, default=0.3)
parser.add_argument("--adversarial-worlds", type=int, default=200)
parser.add_argument("--json", default="")
args = parser.parse_args()

torch.set_num_threads(1)
torch.manual_seed(args.seed)

SLOTS, VALUES = 6, 8
ABSENT = VALUES
HEIGHT = WIDTH = 8
PLANES = 3
PAR_OPS = ("NOOP", "INC", "DEC", "CINC", "CDEC", "COPY", "SINC", "SDEC")
SEQ_OPS = ("NOOP", "INC", "DEC", "CINC", "CDEC", "COPY", "SWAP",
           "SINC", "SDEC")
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


def seq_instruction(state, op, i, j, m):
    name, mod = SEQ_OPS[op], MODULI[m]
    out = state.clone()
    if name == "INC":
        out[:, i] = (state[:, i] + 1) % mod
    elif name == "DEC":
        out[:, i] = (state[:, i] - 1) % mod
    elif name == "SINC":
        out[:, i] = torch.clamp(state[:, i] + 1, max=mod - 1)
    elif name == "SDEC":
        out[:, i] = torch.clamp(state[:, i] - 1, min=0)
    elif name == "CINC":
        g = state[:, j] != 0
        out[:, i] = torch.where(g, (state[:, i] + 1) % mod, state[:, i])
    elif name == "CDEC":
        g = state[:, j] != 0
        out[:, i] = torch.where(g, (state[:, i] - 1) % mod, state[:, i])
    elif name == "COPY":
        out[:, i] = state[:, j]
    elif name == "SWAP":
        out[:, i], out[:, j] = state[:, j], state[:, i]
    return out


def per_slot_search(before, after, only=None):
    program, cost = [], 0
    for s in range(SLOTS):
        if only is not None and s not in only:
            program.append(None)
            continue
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


def infer_moduli(*tensors):
    stacked = torch.cat(list(tensors), dim=0)
    out = []
    for slot in range(SLOTS):
        column = stacked[:, slot]
        column = column[column < VALUES]
        want = (int(column.max()) + 1) if column.numel() else VALUES
        out.append(min(m for m in MODULI if m >= max(want, MODULI[0])))
    return out


def seq_search(before, after, used, depth):
    moduli = infer_moduli(before, after)
    writes = set((before != after).any(dim=0).nonzero().flatten().tolist())
    singles = [(op, i, j, MODULI.index(moduli[i]))
               for op in range(len(SEQ_OPS)) if SEQ_OPS[op] != "NOOP"
               for i in sorted(writes or {0}) for j in range(SLOTS) if j != i]
    best, best_score, cost = [], -1.0, 0
    for one in singles:
        cost += 1
        got = seq_instruction(before, *one)
        score = float((got[:, used] == after[:, used]).float().mean())
        if score > best_score:
            best, best_score = [one], score
        if best_score >= 1.0:
            return best, cost
    if depth >= 2:
        for one in singles:
            mid = seq_instruction(before, *one)
            for two in singles:
                cost += 1
                got = seq_instruction(mid, *two)
                score = float((got[:, used] == after[:, used]).float().mean())
                if score > best_score:
                    best, best_score = [one, two], score
                if best_score >= 1.0:
                    return best, cost
    return best, cost


def run_sequential(state, program):
    out = state
    for one in program:
        out = seq_instruction(out, *one)
    return out


HYBRID_THRESHOLDS = (0.90, 0.95, 0.98)


def hybrid_search(fit_before, fit_after, used,
                  threshold=0.98):
    """Attack 1's own fix. Try the PARALLEL language first, because it is
    flat in arity and cheap; if what it finds does not reproduce the
    evidence, fall back to sequential depth-2, which can express the data
    dependence parallel cannot.

    The choice is made on the FIT rows only -- picking whichever scores
    better on the held-out rows would be choosing with the answer.
    Returns (executor, program, cost, used_fallback)."""
    par, cost = per_slot_search(fit_before, fit_after)
    par_fit = float((run_parallel(fit_before, par)[:, used]
                     == fit_after[:, used]).float().mean())
    if par_fit >= threshold:
        return run_parallel, par, cost, False
    seq, spent = seq_search(fit_before, fit_after, used, 2)
    cost += spent
    seq_fit = float((run_sequential(fit_before, seq)[:, used]
                     == fit_after[:, used]).float().mean())
    if seq_fit > par_fit:
        return run_sequential, seq, cost, True
    return run_parallel, par, cost, False


report = {"seed": args.seed}
ALL = torch.ones(SLOTS, dtype=torch.bool)

# =====================================================================
# ATTACK 1 -- worlds chosen so the parallel language CANNOT express them
# =====================================================================
def adversarial_program(generator, kind: str):
    """Sequential depth-2 programs with a true data dependence.

    `chain`  -- the second instruction READS the slot the first WROTE,
                so its input does not exist in the pre-state at all.
    `twice`  -- the same slot incremented twice, i.e. +2, which no
                single per-slot write in this vocabulary produces.
    """
    i = int(torch.randint(0, SLOTS, (1,), generator=generator))
    m = MODULI.index(VALUES)
    if kind == "twice":
        return [(SEQ_OPS.index("INC"), i, (i + 1) % SLOTS, m),
                (SEQ_OPS.index("INC"), i, (i + 1) % SLOTS, m)]
    b = int(torch.randint(0, SLOTS, (1,), generator=generator))
    if b == i:
        b = (b + 1) % SLOTS
    first = (SEQ_OPS.index("INC"), i, (i + 1) % SLOTS, m)
    second = (SEQ_OPS.index("COPY"), b, i, m)     # reads the NEW slot i
    return [first, second]


attack1 = {}
gen1 = torch.Generator().manual_seed(args.seed + 11)
for kind in ("chain", "twice"):
    par, seq2, seq1 = [], [], []
    hyb = {t: [] for t in HYBRID_THRESHOLDS}
    hyb_cost = {t: [] for t in HYBRID_THRESHOLDS}
    hyb_fall = {t: [] for t in HYBRID_THRESHOLDS}
    for _ in range(args.adversarial_worlds):
        program = adversarial_program(gen1, kind)
        fit = torch.randint(0, VALUES, (args.examples, SLOTS), generator=gen1)
        held = torch.randint(0, VALUES, (args.eval_rows, SLOTS),
                             generator=gen1)
        fit_after = run_sequential(fit, program)
        held_after = run_sequential(held, program)
        prog, _ = per_slot_search(fit, fit_after)
        par.append(float((run_parallel(held, prog) == held_after)
                         .float().mean()))
        for t in HYBRID_THRESHOLDS:
            run, found, spent, fell = hybrid_search(fit, fit_after, ALL, t)
            hyb[t].append(float((run(held, found) == held_after)
                                .float().mean()))
            hyb_cost[t].append(spent)
            hyb_fall[t].append(int(fell))
        for depth, sink in ((1, seq1), (2, seq2)):
            found, _ = seq_search(fit, fit_after, ALL, depth)
            sink.append(float((run_sequential(held, found) == held_after)
                              .float().mean()))
    attack1[kind] = {
        "n": len(par),
        "parallel": round(sum(par) / len(par), 4),
        "sequential_depth1": round(sum(seq1) / len(seq1), 4),
        "sequential_depth2": round(sum(seq2) / len(seq2), 4),
        "HYBRID": {str(t): {
            "fit": round(sum(hyb[t]) / len(hyb[t]), 4),
            "candidates": round(sum(hyb_cost[t]) / len(hyb_cost[t]), 1),
            "fallback_rate": round(sum(hyb_fall[t]) / len(hyb_fall[t]), 4)}
            for t in HYBRID_THRESHOLDS}}
report["attack1_language_on_adversarial_worlds"] = attack1


# =====================================================================
# the plant and the reader, built exactly as F206 recommends
# =====================================================================
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


def coupled_parallel(generator):
    """ATTACK 5: every slot written by an instruction that READS another
    slot, no NOOPs. The hardest programs in the language for a plant
    that has to keep six writes straight at once."""
    hard = [PAR_OPS.index(n) for n in ("CINC", "CDEC", "COPY")]
    out = []
    for s in range(SLOTS):
        op = hard[int(torch.randint(0, len(hard), (1,), generator=generator))]
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
    loss = torch.nn.functional.cross_entropy(
        interp(prog, st).reshape(-1, VALUES),
        run_parallel(st, prog).reshape(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
for parameter in interp.parameters():
    parameter.requires_grad_(False)


def plant_gate(sampler, seed):
    generator = torch.Generator().manual_seed(seed)
    hits = rows = 0
    for _ in range(64):
        prog = sampler(generator)
        st = torch.randint(0, VALUES, (128, SLOTS), generator=generator)
        with torch.no_grad():
            hits += int((interp(prog, st).argmax(-1)
                         == run_parallel(st, prog)).sum())
        rows += st.numel()
    return round(hits / rows, 4)


report["attack5_plant"] = {
    "random_programs": plant_gate(random_parallel, args.seed + 5551),
    "maximally_coupled_programs": plant_gate(coupled_parallel,
                                             args.seed + 5551)}


# ------------------------------------------------------------- worlds
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


def wake(count, generator):
    """As F206, but the ACTION is recorded so a per-action baseline can
    be built from exactly the same evidence the reader trained on."""
    primitives = family_variants()[:args.wake_games]
    rows, guard = [], 0
    while len(rows) < count:
        guard += 1
        if guard > count * 50:
            raise SystemExit("wake phase could not fill the pool")
        if float(torch.rand(1, generator=generator)) < args.synthetic_share:
            prog = random_parallel(generator)
            before = torch.randint(0, VALUES, (args.examples, SLOTS),
                                   generator=generator)
            after = run_parallel(before, prog)
            rows.append((before, after, per_slot_search(before, after)[0], -1))
            continue
        draw = 2 * args.examples
        if len(rows) % 2 == 0:
            config = primitives[int(torch.randint(
                0, len(primitives), (1,), generator=generator))]
            seed = int(torch.randint(0, 10 ** 6, (1,), generator=generator))
            action = int(torch.randint(0, 4, (1,), generator=generator))
            pair = game_transitions(config, draw, seed, action)
        else:
            family = RandomFamily(random_family_spec(generator))
            action = -1
            pair = family_transitions(
                family,
                draw,
                int(torch.randint(0, family.actions, (1,),
                                  generator=generator)),
                generator)
        clean = sanitise(*pair)
        if clean is None or clean[0].shape[0] < args.examples:
            continue
        rows.append((clean[0][:args.examples], clean[1][:args.examples],
                     per_slot_search(clean[0][:args.examples],
                                     clean[1][:args.examples])[0], action))
    return (torch.stack([r[0] for r in rows]),
            torch.stack([r[1] for r in rows]),
            torch.tensor([[list(t) for t in r[2]] for r in rows]),
            torch.tensor([r[3] for r in rows]))


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


pool = wake(args.pool, torch.Generator().manual_seed(args.seed * 15485863))
reader = Reader(args.dim)
r_opt = torch.optim.AdamW(reader.parameters(), lr=args.lr,
                          weight_decay=args.weight_decay)
gen = torch.Generator().manual_seed(args.seed * 104729)
for _ in range(args.reader_updates):
    pick = torch.randint(0, pool[0].shape[0], (args.batch_size,),
                         generator=gen)
    before, after, labels = pool[0][pick], pool[1][pick], pool[2][pick]
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


def read_program(before, after):
    with torch.no_grad():
        po, pj, pm = reader(before.unsqueeze(0), after.unsqueeze(0))
    return [(int(po[0, s].argmax()), int(pj[0, s].argmax()),
             int(pm[0, s].argmax())) for s in range(SLOTS)]


# ---------------------------------------------- ATTACK 2 baselines
# Per-action mode: the program the search most often returns for THIS
# ACTION INDEX across the worlds the reader trained on. On grid worlds
# every action has the same avatar semantics everywhere, so this is a
# genuinely strong competitor rather than a floor.
action_mode = {}
for act in range(4):
    counts: dict = {}
    for row in range(pool[2].shape[0]):
        if int(pool[3][row]) != act:
            continue
        key = tuple(tuple(int(v) for v in pool[2][row, s])
                    for s in range(SLOTS))
        counts[key] = counts.get(key, 0) + 1
    if counts:
        action_mode[act] = [tuple(t) for t in
                            max(counts.items(), key=lambda kv: kv[1])[0]]

# Nearest neighbour: retrieval with NO learning, over the same pool the
# reader was trained on, keyed by the same features the reader sees.
POOL_KEY = torch.cat([
    torch.nn.functional.one_hot(pool[0], VALUES).float().mean(dim=1)
    .view(pool[0].shape[0], -1),
    torch.nn.functional.one_hot(pool[1], VALUES).float().mean(dim=1)
    .view(pool[1].shape[0], -1)], dim=-1)


def nearest_program(before, after):
    key = torch.cat([
        torch.nn.functional.one_hot(before, VALUES).float().mean(dim=0)
        .view(-1),
        torch.nn.functional.one_hot(after, VALUES).float().mean(dim=0)
        .view(-1)])
    index = int((POOL_KEY - key).pow(2).sum(dim=-1).argmin())
    return [tuple(int(v) for v in pool[2][index, s]) for s in range(SLOTS)]


# ------------------------------------------------------ the evaluation
def collect(section):
    draws = []
    if section == "held_out_games":
        for config in family_variants()[args.wake_games:]:
            for action in range(4):
                draws.append((
                    action,
                    game_transitions(config, args.eval_rows, 90001 + action,
                                     action),
                    game_transitions(config, args.eval_rows, 777001 + action,
                                     action)))
    else:
        held_gen = torch.Generator().manual_seed(args.seed + 31337)
        for _ in range(120):
            family = RandomFamily(random_family_spec(held_gen))
            action = int(torch.randint(0, family.actions, (1,),
                                       generator=held_gen))
            draws.append((
                -1,
                family_transitions(family, args.eval_rows, action, held_gen),
                family_transitions(family, args.eval_rows, action, held_gen)))
    return draws


def permute(pair, order):
    return pair[0][:, order], pair[1][:, order]


sections = {}
for name in ("held_out_games", "held_out_rule_families"):
    rows = []
    perm_gen = torch.Generator().manual_seed(args.seed + 909)
    for action, fit_pair, held_pair in collect(name):
        fit, held = sanitise(*fit_pair), sanitise(*held_pair)
        if fit is None or held is None:
            continue
        fb, fa, _ = fit
        hb, ha, used = held
        if int(used.sum()) == 0 or hb.shape[0] < 8:
            continue
        moving = (hb != ha).any(dim=0) & used
        if int(moving.sum()) == 0:
            continue

        def fit_of(program):
            return float((run_parallel(hb, program)[:, moving]
                          == ha[:, moving]).float().mean())

        fbe, fae = fb[:args.examples], fa[:args.examples]
        row = {
            "identity": float((hb[:, moving] == ha[:, moving]).float().mean()),
            "reader": fit_of(read_program(fbe, fae)),
            "search": fit_of(per_slot_search(fbe, fae)[0]),
            "nearest_neighbour": fit_of(nearest_program(fbe, fae))}
        for t in HYBRID_THRESHOLDS:
            hrun, hprog, hcost, hfell = hybrid_search(fbe, fae, moving, t)
            row[f"hybrid{t}"] = float((hrun(hb, hprog)[:, moving]
                                       == ha[:, moving]).float().mean())
            row[f"hybrid{t}_cost"] = hcost
            row[f"hybrid{t}_fallback"] = int(hfell)
        if action in action_mode:
            row["per_action_mode"] = fit_of(action_mode[action])

        # ATTACK 3: permute the slots. A legitimate new world; the
        # per-slot search is equivariant and must be unaffected.
        order = torch.randperm(SLOTS, generator=perm_gen)
        pfb, pfa = permute((fb, fa), order)
        phb, pha = permute((hb, ha), order)
        pmoving = moving[order]

        def pfit_of(program):
            return float((run_parallel(phb, program)[:, pmoving]
                          == pha[:, pmoving]).float().mean())
        row["reader_permuted"] = pfit_of(
            read_program(pfb[:args.examples], pfa[:args.examples]))
        row["search_permuted"] = pfit_of(
            per_slot_search(pfb[:args.examples], pfa[:args.examples])[0])

        # ATTACK 4: starve the evidence.
        for budget in (4, 8, 16):
            row[f"reader_at{budget}"] = fit_of(read_program(fb[:budget],
                                                            fa[:budget]))
            row[f"search_at{budget}"] = fit_of(
                per_slot_search(fb[:budget], fa[:budget])[0])
        rows.append(row)
    sections[name] = rows

summary = {}
for name, rows in sections.items():
    def mean(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None
    summary[name] = {
        "n": len(rows),
        "attack2_hostile_baselines": {
            "identity_floor": mean("identity"),
            "per_action_mode": mean("per_action_mode"),
            "nearest_neighbour_retrieval": mean("nearest_neighbour"),
            "READER": mean("reader"),
            "search": mean("search"),
            "hybrid": {str(t): {
                "fit": mean(f"hybrid{t}"),
                "candidates": mean(f"hybrid{t}_cost"),
                "fallback_rate": mean(f"hybrid{t}_fallback")}
                for t in HYBRID_THRESHOLDS}},
        "attack3_slot_permutation": {
            "reader": mean("reader"), "reader_permuted": mean("reader_permuted"),
            "search": mean("search"),
            "search_permuted": mean("search_permuted")},
        "attack4_starved_evidence": {
            f"{b}": {"reader": mean(f"reader_at{b}"),
                     "search": mean(f"search_at{b}")}
            for b in (4, 8, 16)}}
    summary[name]["attack4_starved_evidence"]["32"] = {
        "reader": mean("reader"), "search": mean("search")}
report["summary"] = summary
report["detail"] = sections

print(json.dumps({k: v for k, v in report.items() if k != "detail"},
                 indent=2))
if args.json:
    with open(args.json, "w") as handle:
        json.dump(report, handle, indent=2)

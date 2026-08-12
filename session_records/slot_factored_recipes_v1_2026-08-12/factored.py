"""Factor the recipe by SLOT, and the search stops mattering.

F202 put a reader in front of the search and it read held-out rule
families at 1.0000. F203 planned with the resulting bank and matched the
true simulator on three of four games. Both stopped at the same wall,
and I named it as arity: **the reader emits ONE instruction, and grid
actions change two or three slots**, so everything above arity 1 still
needed enumeration.

The wall is not arity. It is SEQUENTIAL SEMANTICS.

A sequential recipe is a list, so its length is unbounded, its search
space is exponential in depth, and a reader for it needs a variable
number of heads. Measured on real grid actions (scratch probe, six
worlds x four actions, 128 rows, best fit on the used-slot block):

    identity floor            0.6022
    depth-1 sequential        0.8298      112 candidates
    depth-2 sequential        0.9414   12,598 candidates
    per-slot PARALLEL         0.9443      776 candidates

A parallel recipe assigns each slot one instruction, every one reading
the PRE-state. It reaches depth-2's fit at a sixteenth of the cost, and
three things fall out at once:

  1. **The search factorises.** Slot s is fit against column s alone, so
     six independent searches of 280 candidates replace one search of
     12,598. Arity stops costing anything -- a program writing three
     slots costs exactly what one writing a single slot costs.
  2. **The output shape is FIXED.** SLOTS x (op, j, m), always. A reader
     needs 18 heads and never needs to decide how long the program is.
  3. **SWAP becomes a derived form** -- SWAP i,j is COPY i<-j at slot i
     together with COPY j<-i at slot j -- so the per-slot vocabulary is
     smaller than the sequential one while expressing more.

None of this is domain-specific: simultaneous assignment is the ordinary
semantics of a state-transition rule, and the plant still learns it from
random programs over random states with no world in sight.

What is measured here:

  * the plant executes parallel programs faithfully (gate, reported);
  * per-slot search fits HELD-OUT transitions, not just the ones it fit;
  * a reader trained on the search's labels emits whole programs in one
    forward pass, scored on worlds it never saw;
  * against the identity floor, a shuffled-label control, and the search
    itself as the ceiling.
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
parser.add_argument("--examples", type=int, default=32,
                    help="transitions the reader sees per action")
parser.add_argument("--eval-rows", type=int, default=128)
parser.add_argument("--pool", type=int, default=4000,
                    help="labelled examples the WAKE phase produces once")
parser.add_argument("--train-source", default="families",
                    choices=("families", "primitives", "mixed"),
                    help="which worlds the reader learns from. `families` "
                         "is the strict arm: only synthetic rule families, "
                         "so every grid game is a different DOMAIN as well "
                         "as a new world. `primitives` uses grid worlds. "
                         "`mixed` uses both.")
parser.add_argument("--wake-games", type=int, default=7,
                    help="how many grid variants the wake phase may use. "
                         "The REST are held out and never seen in any "
                         "form. Seven is the seven single-mechanic games, "
                         "which is a harder split than it sounds: the "
                         "eighteen compounds populate a SECOND object "
                         "plane, so slots 4 and 5 never move in training. "
                         "Raise it to give the reader worlds of the shape "
                         "it will be tested on while still holding the "
                         "specific worlds out.")
parser.add_argument("--synthetic-share", type=float, default=0.0,
                    help="fraction of the wake pool drawn as SYNTHETIC "
                         "worlds: a random parallel program over random "
                         "states, labelled by the same per-slot search. "
                         "The reason to want this is a counting argument. "
                         "Fifteen grid worlds times four actions is sixty "
                         "distinct grid labels, which a reader can "
                         "memorise, and that is exactly the shape of the "
                         "result without it -- equal to the search on "
                         "world-shapes it has seen, action-level only on "
                         "new ones. Synthetic worlds are unlimited and "
                         "carry arity up to six. F201 measured that random "
                         "STATES cost 0.41 of functional accuracy on real "
                         "families, so this is expected to trade real-world "
                         "fit for generalisation, and the point is to find "
                         "out at what rate.")
parser.add_argument("--shuffle-labels", action="store_true",
                    help="control: labels paired with the WRONG "
                         "transitions. A reader that has only learned "
                         "which programs are COMMON scores above the floor "
                         "here without having read anything.")
parser.add_argument("--language-compare", action="store_true",
                    help="also fit SEQUENTIAL depth-1 and depth-2 programs "
                         "on the same evidence and score them on the same "
                         "held-out rows. Off by default because depth-2 "
                         "costs about 12,000 candidates per world, which "
                         "is the point being measured.")
parser.add_argument("--no-plant", action="store_true",
                    help="score with ground-truth execution instead of the "
                         "neural plant, to separate reader error from "
                         "plant error")
parser.add_argument("--json", default="")
args = parser.parse_args()

torch.set_num_threads(1)
torch.manual_seed(args.seed)

SLOTS, VALUES = 6, 8
ABSENT = VALUES
HEIGHT = WIDTH = 8
PLANES = 3

# SWAP is absent on purpose: under parallel semantics it is COPY i<-j at
# slot i together with COPY j<-i at slot j, so keeping it would put two
# spellings of one function in the search space and in the reader's
# label distribution.
PAR_OPS = ("NOOP", "INC", "DEC", "CINC", "CDEC", "COPY", "SINC", "SDEC")
MODULI = tuple(range(2, VALUES + 1))
NOOP = (0, 0, 0)
REPAIR_THRESHOLDS = (0.80, 0.90, 0.95, 0.98, 1.00)


def slot_write(state, s, op, j, m):
    """The value written into slot s. Reads only the PRE-state, which is
    what makes the six slot updates independent of each other and of
    their order."""
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
    """program: SLOTS triples (op, j, m). All applied simultaneously."""
    out = state.clone()
    for s in range(SLOTS):
        out[:, s] = slot_write(state, s, *program[s])
    return out


# ------------------------------- the language this one is measured against
# The sequential ISA every finding from F158 to F203 used, kept here so
# the two languages can be fit on the SAME evidence and scored on the
# SAME held-out rows. Executed by ground truth on both sides, because the
# question is which language expresses the world, not which plant runs it.
SEQ_OPS = ("NOOP", "INC", "DEC", "CINC", "CDEC", "COPY", "SWAP",
           "SINC", "SDEC")


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
    """cross_domain's search, unchanged in kind: effect-restricted whole
    programs scored on the whole used-slot block."""
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


# ----------------------------------------------------------- the plant
class Interpreter(torch.nn.Module):
    """Program + state -> next state, one parallel update per call.

    Two details decide whether this works at all, and both were measured
    rather than assumed (scratch sweep, 12,000 updates):

        program code SUMMED, 1 refinement step      0.3747
        program code SUMMED, 3 refinement steps     0.3809
        program code SUMMED, 6 refinement steps     0.4055
        one step per slot, PRE-STATE re-fed         1.0000

    Summing the six per-slot codes into one vector asks the network to
    unbind six superposed instructions, and it does not; depth does not
    rescue it (0.3747 -> 0.4055 for six times the compute). Folding one
    slot code per residual step does, and lands exactly.

    The pre-state `base` is re-supplied at every step because that is
    what parallel semantics means -- every write reads the state as it
    was before any of them ran. I expected that to be the reason this
    plant is exact where the sequential one reached 0.9896, and the
    ABLATION SAYS OTHERWISE: replacing `base` with the running latent
    also scores 1.0000. Per-slot decomposition is the whole effect, and
    the network carries the original values itself. `base` is kept
    because it makes the semantics explicit, not because it earns its
    place in the measurement."""

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
          "shuffle_labels": args.shuffle_labels,
          "interpreter_updates": args.interpreter_updates,
          "interpreter_check": round(hits / rows, 4)}


# ------------------------------------------------------- perception
def slot_state(screen: torch.Tensor) -> torch.Tensor:
    """Screen -> slots. Unchanged from `cross_domain.py`: argmax for the
    avatar, nearest object by Manhattan distance per plane."""
    frames = screen.view(-1, PLANES, HEIGHT, WIDTH)
    batch = frames.shape[0]
    out = torch.full((batch, SLOTS), ABSENT, dtype=torch.long)
    rows_ix = torch.arange(HEIGHT).view(-1, 1).expand(HEIGHT, WIDTH)
    cols_ix = torch.arange(WIDTH).view(1, -1).expand(HEIGHT, WIDTH)
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
            distance = (rows_ix - ar).abs() + (cols_ix - ac).abs()
            distance = torch.where(mask, distance,
                                   torch.full_like(distance, 999))
            pick = int(distance.reshape(-1).argmin())
            out[row, base], out[row, base + 1] = pick // WIDTH, pick % WIDTH
    return out


def sanitise(before, after):
    """F192's row/slot split. A missing AVATAR means the episode ended
    and the row has no successor, so drop the ROW; a missing OBJECT means
    the world does not use that slot, so mask the SLOT and keep the row."""
    alive = (before[:, 0] < VALUES) & (after[:, 0] < VALUES)
    before, after = before[alive], after[alive]
    if before.shape[0] == 0:
        return None
    used = (before < VALUES).all(dim=0) & (after < VALUES).all(dim=0)
    before = torch.where(before < VALUES, before, torch.zeros_like(before))
    after = torch.where(after < VALUES, after, torch.zeros_like(after))
    return before, after, used


def game_transitions(config, count: int, seed: int, action: int):
    verifier = FamilyVerifier(config, batch_size=count, seed=seed)
    verifier.reset(seed=seed)
    before = slot_state(verifier.observation())
    verifier.step(torch.full((count,), action, dtype=torch.long))
    return before, slot_state(verifier.observation())


def family_transitions(family, count: int, action: int, generator):
    size = len(family.states)
    idx = torch.randint(0, size, (count,), generator=generator)
    nxt = torch.tensor([family.table[int(x)][action] for x in idx])
    return family.slot_values(idx), family.slot_values(nxt)


# ------------------------------------------------------ the wake phase
def per_slot_search(before, after, only=None):
    """Six independent searches. Slot s is fit against column s alone, so
    what a program does to the other five cannot make it look better or
    worse here. Returns the program and the number of candidates tried.

    `only` restricts the search to a subset of slots, which is what makes
    REPAIR possible: the reader proposes a whole program, each slot is
    checked in one pass, and enumeration is spent only where the check
    failed."""
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


def wake(count: int, generator, source: str):
    """Solve worlds and keep what the solution was. Rule families are
    drawn fresh; grid games are drawn from the seven primitives, whose
    eighteen compounds are never touched here."""
    primitives = family_variants()[:args.wake_games]
    rows, guard = [], 0
    while len(rows) < count:
        guard += 1
        if guard > count * 50:
            raise SystemExit("wake phase could not fill the pool")
        if (args.synthetic_share > 0
                and float(torch.rand(1, generator=generator))
                < args.synthetic_share):
            # A synthetic world: a random parallel program over random
            # states. Labelled by the SAME search as everything else, not
            # by the program that generated it, because two programs can
            # agree on a batch and the search's answer is what the reader
            # is being taught to predict.
            prog = random_parallel(generator)
            before = torch.randint(0, VALUES, (args.examples, SLOTS),
                                   generator=generator)
            after = run_parallel(before, prog)
            rows.append((before, after, per_slot_search(before, after)[0]))
            continue
        take_game = (source == "primitives" or
                     (source == "mixed" and len(rows) % 2 == 0))
        # Over-draw: a grid episode can END, and `sanitise` drops those
        # rows, so asking for exactly `examples` returns a ragged pool.
        draw = 2 * args.examples
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
    before = torch.stack([r[0] for r in rows])
    after = torch.stack([r[1] for r in rows])
    labels = torch.tensor([[list(t) for t in r[2]] for r in rows])
    return before, after, labels          # labels: (count, SLOTS, 3)


# ----------------------------------------------------------- the reader
class Reader(torch.nn.Module):
    """Transitions -> a whole parallel program, in one forward pass.

    A set encoder over (state, next state) pairs, then SLOTS x 3 heads.
    The output shape does not depend on how many slots the world's action
    actually writes, which is the entire reason this can cover arity
    above one where F202's reader could not."""

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


pool = wake(args.pool, torch.Generator().manual_seed(args.seed * 15485863),
            args.train_source)
report["pool"] = int(pool[0].shape[0])

reader = Reader(args.dim)
r_opt = torch.optim.AdamW(reader.parameters(), lr=args.lr,
                          weight_decay=args.weight_decay)
gen = torch.Generator().manual_seed(args.seed * 104729)
curve = []
for update in range(args.reader_updates):
    pick = torch.randint(0, pool[0].shape[0], (args.batch_size,),
                         generator=gen)
    before, after, labels = (t[pick] for t in pool)
    if args.shuffle_labels:
        perm = torch.randperm(args.batch_size, generator=gen)
        labels = labels[perm]
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
    if update % 500 == 0:
        curve.append((update, round(float(loss), 4)))
report["reader_curve"] = curve


# A SECOND faithfulness gate, on the states that actually matter. The
# gate above uses random states and random programs, which is the
# training distribution; a plant can be exact there and drift on the
# structured states real worlds produce, and that failure would show up
# as "the reader is wrong" rather than "the plant is wrong".
plant_hits = plant_rows = 0
for _row in range(min(400, pool[0].shape[0])):
    _prog = [tuple(int(v) for v in pool[2][_row, _s]) for _s in range(SLOTS)]
    _st = pool[0][_row]
    with torch.no_grad():
        _got = interp(_prog, _st).argmax(-1)
    plant_hits += int((_got == run_parallel(_st, _prog)).sum())
    plant_rows += _st.numel()
report["interpreter_check_real_states"] = round(plant_hits / plant_rows, 4)

# The most common program in the wake pool, applied to every world
# regardless of what its transitions say. A reader that had learned only
# which programs are FREQUENT would score exactly this, so anything the
# reader gains over it is gained by reading.
_counts: dict = {}
for _row in range(pool[2].shape[0]):
    _key = tuple(tuple(int(v) for v in pool[2][_row, _s])
                 for _s in range(SLOTS))
    _counts[_key] = _counts.get(_key, 0) + 1
MODE_PROGRAM = [tuple(t) for t in max(_counts.items(),
                                      key=lambda kv: kv[1])[0]]
report["mode_program"] = [list(t) for t in MODE_PROGRAM]
report["mode_program_share"] = round(max(_counts.values())
                                     / pool[2].shape[0], 4)


def read_program(before, after):
    with torch.no_grad():
        po, pj, pm = reader(before.unsqueeze(0), after.unsqueeze(0))
    return [(int(po[0, s].argmax()), int(pj[0, s].argmax()),
             int(pm[0, s].argmax())) for s in range(SLOTS)]


def execute(program, state):
    if args.no_plant:
        return run_parallel(state, program)
    with torch.no_grad():
        return interp(program, state).argmax(-1)


# ------------------------------------------------------- the evaluation
# Every score below is on transitions NEITHER the reader nor the search
# ever saw: the program is chosen from one draw of the world and scored
# on a different draw. Fitting and scoring the same rows would make a
# per-slot search look perfect by construction, since it has six free
# choices to spend on six columns.
def evaluate_world(fit_pair, held_pair, wrong_pair=None,
                   wrong_action_pair=None):
    fit = sanitise(*fit_pair)
    held = sanitise(*held_pair)
    if fit is None or held is None:
        return None
    fb, fa, _ = fit
    hb, ha, used = held
    if int(used.sum()) == 0 or hb.shape[0] < 8:
        return None
    # SAME EVIDENCE for both arms. The question is not whether a search
    # with unlimited data beats a reader with little, it is whether the
    # reader can do what the search does from what the search is given.
    searched, cost = per_slot_search(fb[:args.examples], fa[:args.examples])
    guessed = read_program(fb[:args.examples], fa[:args.examples])

    # REPAIR: the architecture's own stated endpoint, that search becomes
    # VERIFICATION rather than discovery. Check each of the reader's six
    # instructions against the column it writes -- one execution per
    # slot, no enumeration -- and re-search only the slots that fail.
    # Where the reader is right this costs six checks instead of a
    # thousand candidates; where it is wrong the search is still there.
    # A THRESHOLD of 1.0 -- repair anything not exact -- is the wrong
    # default in a noisy domain: no grid column is exactly reproducible,
    # so every slot is re-searched and nothing is saved. Sweeping the
    # threshold traces the frontier between one forward pass and the full
    # search, which is the useful object.
    fbe, fae = fb[:args.examples], fa[:args.examples]
    checked = [float((slot_write(fbe, s, *guessed[s]) == fae[:, s])
                     .float().mean()) for s in range(SLOTS)]
    repairs = {}
    for threshold in REPAIR_THRESHOLDS:
        bad = {s for s in range(SLOTS) if checked[s] < threshold}
        spent = SLOTS                        # one execution per slot
        fixed = list(guessed)
        if bad:
            patch, extra = per_slot_search(fbe, fae, only=bad)
            spent += extra
            for s in bad:
                fixed[s] = patch[s]
        repairs[threshold] = (fixed, spent, len(bad))

    # The used-slot block is padded with slots the action never touches,
    # and every arm including the identity gets those for free. `moving`
    # is the sharp version: only the slots that actually change.
    moving = (hb != ha).any(dim=0) & used

    def fit_of(program, mask):
        if int(mask.sum()) == 0:
            return None
        got = execute(program, hb)
        return float((got[:, mask] == ha[:, mask]).float().mean())

    out = {"search": fit_of(searched, used), "reader": fit_of(guessed, used),
           "identity": float((hb[:, used] == ha[:, used]).float().mean()),
           "search_moving": fit_of(searched, moving),
           "reader_moving": fit_of(guessed, moving),
           "mode_moving": fit_of(MODE_PROGRAM, moving),
           "identity_moving": (float((hb[:, moving] == ha[:, moving])
                                     .float().mean())
                               if int(moving.sum()) else None),
           "search_cost": cost, "reader_cost": 1,
           **{f"repair{t}": fit_of(p, used) for t, (p, _, _) in repairs.items()},
           **{f"repair{t}_moving": fit_of(p, moving)
              for t, (p, _, _) in repairs.items()},
           **{f"repair{t}_cost": c for t, (_, c, _) in repairs.items()},
           **{f"repair{t}_slots": n for t, (_, _, n) in repairs.items()},
           "agreement": float(sum(int(searched[s] == guessed[s])
                                  for s in range(SLOTS)) / SLOTS),
           "changed": int(moving.sum())}
    # WRONG-WORLD control, at inference rather than training time. The
    # shuffled-label arm asks whether the reader could learn without
    # reading; this asks whether the TRAINED reader is using the
    # transitions in front of it, by handing it another world's and
    # scoring the answer here. A reader keying on anything other than its
    # input scores the same either way.
    #
    # TWO variants, because on grid worlds they mean different things.
    # Every grid game shares its avatar dynamics: action 2 moves the
    # avatar the same way in `collect1` and in `avoid2`, and only the
    # OBJECT behaviour differs. So a control that swaps the world but
    # keeps the ACTION hands the reader most of the right answer, and a
    # reader scoring level with it has still read the action correctly --
    # it has only failed to read what is specific to this world. Swapping
    # both is the control for reading anything at all.
    for tag, pair in (("wrong_world", wrong_pair),
                      ("wrong_world_and_action", wrong_action_pair)):
        if pair is None:
            continue
        other = sanitise(*pair)
        if other is not None and other[0].shape[0] >= args.examples:
            stray = read_program(other[0][:args.examples],
                                 other[1][:args.examples])
            out[f"reader_{tag}"] = fit_of(stray, used)
            out[f"reader_{tag}_moving"] = fit_of(stray, moving)
    if args.language_compare:
        fbe, fae = fb[:args.examples], fa[:args.examples]
        # Ground-truth execution on BOTH sides, so this compares
        # languages rather than plants.
        #
        # In-sample fit is reported next to held-out fit because the
        # obvious objection to the parallel language is CAPACITY: it
        # picks six instructions where sequential depth-2 picks two, so
        # of course it fits better. The answer has to be measured, not
        # argued -- if the extra freedom is bought by overfitting, the
        # gap between the rows it was fit on and the rows it was not
        # will be wider for parallel than for sequential.
        out["lang_parallel"] = float(
            (run_parallel(hb, searched)[:, used] == ha[:, used])
            .float().mean())
        out["lang_parallel_insample"] = float(
            (run_parallel(fbe, searched)[:, used] == fae[:, used])
            .float().mean())
        for depth in (1, 2):
            prog, spent = seq_search(fbe, fae, used, depth)
            out[f"lang_seq{depth}"] = float(
                (run_sequential(hb, prog)[:, used] == ha[:, used])
                .float().mean())
            out[f"lang_seq{depth}_insample"] = float(
                (run_sequential(fbe, prog)[:, used] == fae[:, used])
                .float().mean())
            out[f"lang_seq{depth}_cost"] = spent
        out["lang_parallel_cost"] = cost
    return out


variants = family_variants()
sections = {}

for label, configs in (("held_out_games", variants[args.wake_games:]),
                       ("seen_games", variants[:args.wake_games])):
    draws = []
    for config in configs:
        for action in range(4):
            draws.append((
                str(getattr(config, "name", config)), action,
                game_transitions(config, args.eval_rows, 90001 + action,
                                 action),
                game_transitions(config, args.eval_rows, 777001 + action,
                                 action)))
    rows = []
    for index, (name, action, fit_pair, held_pair) in enumerate(draws):
        # +4 lands on the same ACTION of a different game; +5 changes
        # both. Each game contributes exactly four consecutive draws.
        wrong = draws[(index + 4) % len(draws)][2]
        wrong_act = draws[(index + 5) % len(draws)][2]
        out = evaluate_world(fit_pair, held_pair, wrong, wrong_act)
        if out is not None:
            rows.append(dict(out, game=name, action=action))
    sections[label] = rows

held_gen = torch.Generator().manual_seed(args.seed + 31337)
draws = []
for _ in range(120):
    family = RandomFamily(random_family_spec(held_gen))
    action = int(torch.randint(0, family.actions, (1,), generator=held_gen))
    draws.append((family_transitions(family, args.eval_rows, action, held_gen),
                  family_transitions(family, args.eval_rows, action,
                                     held_gen)))
rows = []
for index, (fit_pair, held_pair) in enumerate(draws):
    # rule families share no action semantics, so both controls are the
    # same thing here; the second is kept for a uniform table
    out = evaluate_world(fit_pair, held_pair,
                         draws[(index + 1) % len(draws)][0],
                         draws[(index + 2) % len(draws)][0])
    if out is not None:
        rows.append(out)
sections["held_out_rule_families"] = rows

summary = {}
for label, rows in sections.items():
    if not rows:
        summary[label] = {"n": 0}
        continue
    def mean(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None
    moving_rows = [r for r in rows if r["changed"] > 0]

    def mean_moving(key):
        vals = [r[key] for r in moving_rows if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None
    summary[label] = {
        "n": len(rows), "identity_floor": mean("identity"),
        "search_fit": mean("search"), "reader_fit": mean("reader"),
        "reader_margin_over_floor": round(mean("reader") - mean("identity"), 4),
        "reader_fraction_of_search": (
            round((mean("reader") - mean("identity"))
                  / (mean("search") - mean("identity")), 4)
            if mean("search") - mean("identity") > 0.01 else None),
        # restricted to slots the action actually moves
        "n_moving": len(moving_rows),
        "identity_floor_moving": mean_moving("identity_moving"),
        "mode_program_moving": mean_moving("mode_moving"),
        "reader_wrong_world_moving": mean_moving("reader_wrong_world_moving"),
        "reader_wrong_world_and_action_moving": mean_moving(
            "reader_wrong_world_and_action_moving"),
        "search_fit_moving": mean_moving("search_moving"),
        "reader_fit_moving": mean_moving("reader_moving"),
        "reader_wrong_world": mean("reader_wrong_world"),
        "exact_program_agreement": mean("agreement"),
        "repair": {str(t): {
            "fit_moving": mean_moving(f"repair{t}_moving"),
            "fit": mean(f"repair{t}"),
            "candidates": mean(f"repair{t}_cost"),
            "slots_repaired": mean(f"repair{t}_slots")}
            for t in REPAIR_THRESHOLDS},
        "search_candidates": mean("search_cost"),
        "reader_forward_passes": 1,
        "mean_changed_slots": mean("changed")}
    if args.language_compare:
        def gap(key):
            held, insample = mean(f"lang_{key}"), mean(f"lang_{key}_insample")
            return (None if held is None or insample is None
                    else round(insample - held, 4))
        summary[label]["language"] = {
            "parallel_fit": mean("lang_parallel"),
            "parallel_cost": mean("lang_parallel_cost"),
            "parallel_overfit_gap": gap("parallel"),
            "sequential_depth1_fit": mean("lang_seq1"),
            "sequential_depth1_cost": mean("lang_seq1_cost"),
            "sequential_depth1_overfit_gap": gap("seq1"),
            "sequential_depth2_fit": mean("lang_seq2"),
            "sequential_depth2_cost": mean("lang_seq2_cost"),
            "sequential_depth2_overfit_gap": gap("seq2")}
report["summary"] = summary
report["detail"] = sections

print(json.dumps(report, indent=2))
if args.json:
    with open(args.json, "w") as handle:
        json.dump(report, handle, indent=2)

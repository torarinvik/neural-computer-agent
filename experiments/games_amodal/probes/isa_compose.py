"""The instruction-set plant, over the AMODAL SLOT INTERFACE.

The boundary this attacks, from the Codex log and reproduced there
from three directions: external context can SELECT existing
computation but cannot INVENT new computation. LITERATURE.md §22-24
frames why our previous probes GUARANTEED that result — we trained the
plant on exactly the two operations the task used, so each was learned
as one opaque function and an activation-space entry could only choose
between them.

**The first version of this probe was DOMAIN SPECIFIC and therefore
wrong.** It gave the plant a bitwise instruction set (AND/OR/XOR/shift
over 8-bit words), which would have produced a bit-manipulation
machine: a plant that could never touch the games, in a project whose
whole premise is one amodal controller. Recorded rather than quietly
replaced, because "solve it with a specialised substrate" is the exact
failure this architecture exists to avoid, and it was one run away
from entering the ledger as progress.

The substrate here is instead the one already shared across every
probe in the project: **SLOTS x VALUES**, the amodal state that
`schema_families.py` uses for procedural rule families and that
`game_slots.py` uses for grid worlds. Encoders map any domain into it;
the plant only ever sees slots. An instruction set over slots is
therefore domain-general by construction — the SAME instructions have
to serve a dial-turning rule family and a foraging grid, because both
are already expressed in these slots.

The instructions are the operations `schema_families.py` established
as its procedural basis, promoted from "how we generate rule families"
to "what the plant executes":

    NOOP                      leave the state alone
    INC  i / DEC  i           advance or retreat one slot, mod VALUES
    CINC i,j / CDEC i,j       the same, CONDITIONAL on slot j
    COPY i,j / SWAP i,j       move content between slots

Conditionals are what make this a basis rather than a lookup table:
without them every program is a fixed permutation of the state, and
with them a program can branch on its own content.

What this probe asks, and only this: **can one shared step function
execute arbitrary programs over that basis, including programs it has
never seen?** Programs are SUPPLIED here, not inferred. Whether a
reader can infer a program from observations is the next question and
deliberately separate — mixing them is how F117 wasted three arms.

If unseen programs execute, then a bank can hold programs, and a new
world-rule is a new arrangement of instructions the plant already
runs — invention becomes composition one level down, without adding a
single domain-specific primitive.
"""

from __future__ import annotations

import argparse
import json

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=69316)
parser.add_argument("--slots", type=int, default=6)
parser.add_argument("--values", type=int, default=8)
parser.add_argument("--dim", type=int, default=128)
parser.add_argument("--train-updates", type=int, default=40000)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight-decay", type=float, default=0.0)
parser.add_argument("--program-len", type=int, default=6)
parser.add_argument(
    "--no-conditionals", action="store_true",
    help="drop CINC/CDEC from the basis. Control: without "
         "conditionals every program is a fixed permutation of the "
         "state, so this measures how much of any success is the "
         "branching rather than the sequencing.")
parser.add_argument(
    "--synthesize", type=int, default=0,
    help="after training the interpreter, SYNTHESISE recipes for real "
         "task families by SEARCH and evaluate them. This is the half "
         "that makes the architecture honest: the interpreter is "
         "trained only on random programs over random states, so no "
         "world ever touches its weights, and a new world is handled "
         "by FINDING a program that explains it rather than by "
         "learning one. Nothing trains during synthesis — the search "
         "proposes candidate programs, scores them with the frozen "
         "interpreter against observed transitions, and keeps the "
         "best. That is the wake phase of DreamCoder and it is also "
         "our own F87 rule: keys address, consequences verify. The "
         "value is the number of candidate programs to try per action.")
parser.add_argument("--observations", type=int, default=64)
parser.add_argument(
    "--library", action="store_true",
    help="LIBRARY LEARNING. Solve families in SEQUENCE, composing "
         "candidate programs from a growing library of FRAGMENTS "
         "rather than from raw instructions, and add each solved "
         "program back as a fragment. F155 found recipes by random "
         "proposal over 252^6 raw programs, which works at length 6 "
         "and cannot scale. The claim to test is the founding "
         "objective itself: if stored programs are reusable, the "
         "search cost for family N should FALL as N grows. The "
         "control is the same sequence with the library frozen at the "
         "primitives — if cost is flat there and falling here, reuse "
         "is doing the work and not the ordering.")
parser.add_argument("--no-growth", action="store_true",
                    help="control: never add fragments to the library")
parser.add_argument(
    "--library-arms", action="store_true",
    help="run the four library policies against ONE shared plant. F157 "
         "measured growth-vs-frozen and found a null, for a reason "
         "stated before the run: fragments were appended to a "
         "UNIFORMLY sampled pool, so the library grew 210 -> 242 and "
         "every useful fragment went from 1-in-210 to 1-in-242. "
         "Accumulation is not a library; COMPRESSION is. This runs "
         "frozen / uniform / prims / weighted in one process so the "
         "arms differ only in proposal policy.")
parser.add_argument(
    "--extra-families", type=int, default=0,
    help="additional procedural families appended to the sequence. "
         "F157's paired test had 7 families and per-family cost spanning "
         "two orders of magnitude, so it had almost no power; reuse can "
         "only pay off on families that come LATER in the sequence.")
parser.add_argument(
    "--related-families", type=int, default=0,
    help="also run the whole arm comparison on a sequence of families "
         "that SHARE their state geometry. The diverse sequence is the "
         "condition where reuse is least able to help by construction, "
         "so a null there alone cannot distinguish a broken library from "
         "tasks with nothing in common.")
parser.add_argument("--fit-target", type=float, default=0.95,
                    help="search stops once a candidate reaches this "
                         "fit, so COST (candidates tried) is the "
                         "measurement rather than final accuracy")
parser.add_argument("--curve-every", type=int, default=0)
parser.add_argument("--json", default="")
args = parser.parse_args()

torch.manual_seed(args.seed)
SLOTS, VALUES = args.slots, args.values

# The basis. Domain-general by construction: these are operations on
# ABSTRACT slots, identical for a rule family and a grid world.
OPS = ("NOOP", "INC", "DEC", "CINC", "CDEC", "COPY", "SWAP")
if args.no_conditionals:
    OPS = tuple(o for o in OPS if o not in ("CINC", "CDEC"))
NOPS = len(OPS)


def run_instruction(state: torch.Tensor, op: int, i: int,
                    j: int) -> torch.Tensor:
    """Ground truth. `state` is (batch, SLOTS) of integers < VALUES."""
    name = OPS[op]
    out = state.clone()
    if name == "NOOP":
        return out
    if name == "INC":
        out[:, i] = (state[:, i] + 1) % VALUES
    elif name == "DEC":
        out[:, i] = (state[:, i] - 1) % VALUES
    elif name == "CINC":
        gate = state[:, j] != 0
        out[:, i] = torch.where(gate, (state[:, i] + 1) % VALUES,
                                state[:, i])
    elif name == "CDEC":
        gate = state[:, j] != 0
        out[:, i] = torch.where(gate, (state[:, i] - 1) % VALUES,
                                state[:, i])
    elif name == "COPY":
        out[:, i] = state[:, j]
    elif name == "SWAP":
        out[:, i], out[:, j] = state[:, j], state[:, i]
    return out


def run_program(program: list, state: torch.Tensor) -> torch.Tensor:
    for op, i, j in program:
        state = run_instruction(state, op, i, j)
    return state


def random_program(generator: torch.Generator, length: int) -> list:
    out = []
    for _ in range(length):
        op = int(torch.randint(0, NOPS, (1,), generator=generator))
        i = int(torch.randint(0, SLOTS, (1,), generator=generator))
        j = int(torch.randint(0, SLOTS, (1,), generator=generator))
        if i == j:
            j = (j + 1) % SLOTS
        out.append((op, i, j))
    return out


class Plant(torch.nn.Module):
    """Task-agnostic core: executes ONE instruction against a latent.

    The instruction is bound once per step rather than attended over —
    F135's result carried across, since that is the interface that
    survived depth. Nothing here mentions bits, grids, or any domain:
    the input is slots and the output is slots.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.load = torch.nn.Linear(SLOTS * VALUES, dim)
        self.op = torch.nn.Embedding(NOPS, dim)
        self.arg_i = torch.nn.Embedding(SLOTS, dim)
        self.arg_j = torch.nn.Embedding(SLOTS, dim)
        self.step = torch.nn.Sequential(
            torch.nn.Linear(2 * dim, 2 * dim), torch.nn.ReLU(),
            torch.nn.Linear(2 * dim, 2 * dim), torch.nn.ReLU(),
            torch.nn.Linear(2 * dim, dim))
        self.norm = torch.nn.LayerNorm(dim)
        self.head = torch.nn.Linear(dim, SLOTS * VALUES)

    def forward(self, program: list, state: torch.Tensor):
        onehot = torch.nn.functional.one_hot(
            state, VALUES).float().view(state.shape[0], -1)
        latent = self.load(onehot)
        for op, i, j in program:
            code = (self.op(torch.tensor(op))
                    + self.arg_i(torch.tensor(i))
                    + self.arg_j(torch.tensor(j)))
            code = code.unsqueeze(0).expand(latent.shape[0], -1)
            # RESIDUAL. Without this the stack is program_len x 3 = 18
            # effective layers with no skip path, and it cannot fit even
            # ONE fixed program: loss pinned at ln(8) = 2.079, i.e.
            # uniform output, flat from update 0 across 40k updates.
            # With the residual the same model fits one program to
            # 1.0000 in 1500 steps.
            latent = self.norm(latent + self.step(
                torch.cat([latent, code], dim=-1)))
        return self.head(latent).view(-1, SLOTS, VALUES)


plant = Plant(args.dim)
optimizer = torch.optim.AdamW(plant.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
data_gen = torch.Generator().manual_seed(args.seed * 15485863)
eval_gen = torch.Generator().manual_seed(args.seed * 7919)
curve: list = []


def evaluate(programs: list, tag: str) -> dict:
    generator = torch.Generator().manual_seed(args.seed * 977)
    slot_hits = exact_hits = rows = 0
    with torch.no_grad():
        for program in programs:
            state = torch.randint(0, VALUES, (256, SLOTS),
                                  generator=generator)
            want = run_program(program, state)
            got = plant(program, state).argmax(-1)
            slot_hits += int((got == want).sum())
            exact_hits += int((got == want).all(dim=-1).sum())
            rows += 256
    return {f"{tag}_slots": round(slot_hits / (rows * SLOTS), 4),
            f"{tag}_exact": round(exact_hits / rows, 4)}


held = [random_program(eval_gen, args.program_len) for _ in range(16)]
held_keys = {tuple(p) for p in held}
longer = [random_program(eval_gen, args.program_len * 2)
          for _ in range(8)]

for update in range(args.train_updates):
    while True:
        program = random_program(data_gen, args.program_len)
        if tuple(program) not in held_keys:
            break
    state = torch.randint(0, VALUES, (args.batch_size, SLOTS),
                          generator=data_gen)
    want = run_program(program, state)
    logits = plant(program, state)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, VALUES), want.reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if args.curve_every and update % args.curve_every == 0:
        curve.append((update, evaluate(held[:4], "h")["h_slots"]))

# chance: the identity-ish prior is not chance, so report BOTH the
# uniform floor and what copying the input unchanged would score
copy_gen = torch.Generator().manual_seed(args.seed * 977)
copy_hits = copy_rows = 0
for program in held:
    state = torch.randint(0, VALUES, (256, SLOTS), generator=copy_gen)
    want = run_program(program, state)
    copy_hits += int((state == want).sum())
    copy_rows += 256
report = {
    "seed": args.seed, "slots": SLOTS, "values": VALUES,
    "program_len": args.program_len, "instructions": list(OPS),
    "no_conditionals": args.no_conditionals,
    "chance_slots": round(1 / VALUES, 4),
    "identity_baseline_slots": round(copy_hits / (copy_rows * SLOTS), 4),
    "curve": curve,
}
report.update(evaluate(held, "held_out_programs"))
report.update(evaluate(longer, "double_length_programs"))

if args.synthesize:
    # Real task families, expressed in the SAME amodal slots. A world's
    # recipe is one program PER ACTION: an action IS a recipe.
    from experiments.games_amodal.probes.schema_families import (
        Family, RandomFamily, random_family_spec)

    def observe(family, count, generator):
        size = len(family.states)
        idx = torch.randint(0, size, (count,), generator=generator)
        act = torch.randint(0, family.actions, (count,),
                            generator=generator)
        nxt = torch.tensor([family.table[int(s)][int(a)]
                            for s, a in zip(idx, act)])
        return family.slot_values(idx), act, family.slot_values(nxt)

    def synthesise(family, generator) -> dict:
        """Search for a program per action. The frozen interpreter is
        the only predictor used, so this measures the interpreter's
        usefulness as a search substrate, not ground-truth fitting."""
        states, acts, nexts = observe(family, args.observations,
                                      generator)
        # A family using fewer than SLOTS slots marks the rest with the
        # sentinel VALUES. Filtering ROWS on that drops every row for
        # such a family; the right move is to mask SLOTS — clamp the
        # unused ones to 0 on input and ignore them when scoring.
        used = (states < VALUES).all(dim=0)
        if int(used.sum()) == 0:
            return {"fit": None, "reason": "no usable slots"}
        states = torch.where(states < VALUES, states,
                             torch.zeros_like(states))
        nexts = torch.where(nexts < VALUES, nexts,
                            torch.zeros_like(nexts))
        recipe, fits = {}, []
        for action in range(family.actions):
            keep = acts == action
            if int(keep.sum()) < 4:
                continue
            src, dst = states[keep], nexts[keep]
            best, best_score = None, -1.0
            for _ in range(args.synthesize):
                candidate = random_program(generator, args.program_len)
                with torch.no_grad():
                    got = plant(candidate, src).argmax(-1)
                score = float((got[:, used] == dst[:, used])
                              .float().mean())
                if score > best_score:
                    best, best_score = candidate, score
            recipe[action] = best
            fits.append(best_score)
        # held-out check: the recipe was chosen on one sample, score it
        # on a fresh one
        held_gen = torch.Generator().manual_seed(args.seed + 4242)
        hs, ha, hn = observe(family, 256, held_gen)
        hs = torch.where(hs < VALUES, hs, torch.zeros_like(hs))
        hn = torch.where(hn < VALUES, hn, torch.zeros_like(hn))
        hits = total = 0
        for action, candidate in recipe.items():
            keep = ha == action
            if not bool(keep.any()):
                continue
            with torch.no_grad():
                got = plant(candidate, hs[keep]).argmax(-1)
            hits += int((got[:, used] == hn[keep][:, used]).sum())
            total += int(keep.sum()) * int(used.sum())
        identity = 0
        for action, candidate in recipe.items():
            keep = ha == action
            if bool(keep.any()):
                identity += int((hs[keep][:, used]
                                 == hn[keep][:, used]).sum())
        return {"search_fit": round(sum(fits) / max(len(fits), 1), 4),
                "held_out": round(hits / max(total, 1), 4),
                "identity": round(identity / max(total, 1), 4)}

    def search_with_library(family, library, weights, generator, budget):
        """Propose programs by concatenating LIBRARY FRAGMENTS. Returns
        the recipe and the number of candidates tried — cost is the
        measurement here, not accuracy.

        `weights` is None for uniform proposal (F157's behaviour) or a
        per-fragment usefulness count. Weighted proposal is the whole
        point of the fix: F157 appended fragments to a UNIFORMLY sampled
        pool, so every added fragment made every other fragment RARER,
        and the library got bigger without getting better."""
        states, acts, nexts = observe(family, args.observations,
                                      generator)
        used = (states < VALUES).all(dim=0)
        states = torch.where(states < VALUES, states,
                             torch.zeros_like(states))
        nexts = torch.where(nexts < VALUES, nexts,
                            torch.zeros_like(nexts))
        # sampling distribution is fixed for the whole family, so build
        # it once rather than per candidate
        table = (torch.tensor(weights, dtype=torch.float)
                 if weights is not None else None)
        recipe, tried_total, fits = {}, 0, []
        for action in range(family.actions):
            keep = acts == action
            if int(keep.sum()) < 4:
                continue
            src, dst = states[keep], nexts[keep]
            best, best_score, tried = None, -1.0, 0
            while tried < budget:
                # build a candidate by concatenating fragments until it
                # is long enough; a fragment may be a whole past recipe
                candidate: list = []
                while len(candidate) < args.program_len:
                    if table is None:
                        slot = int(torch.randint(
                            0, len(library), (1,), generator=generator))
                    else:
                        slot = int(torch.multinomial(
                            table, 1, generator=generator))
                    candidate = candidate + list(library[slot])
                candidate = candidate[:args.program_len]
                tried += 1
                with torch.no_grad():
                    got = plant(candidate, src).argmax(-1)
                score = float((got[:, used] == dst[:, used])
                              .float().mean())
                if score > best_score:
                    best, best_score = candidate, score
                if best_score >= args.fit_target:
                    break
            recipe[action] = best
            fits.append(best_score)
            tried_total += tried
        return recipe, tried_total, sum(fits) / max(len(fits), 1)

    def occurrences(fragment: tuple, program: tuple) -> int:
        """How many times `fragment` appears CONTIGUOUSLY in `program`."""
        width = len(fragment)
        return sum(1 for start in range(len(program) - width + 1)
                   if program[start:start + width] == fragment)

    def sub_fragments(program: tuple) -> list:
        """Every contiguous sub-program of length 2 or more. Whole-recipe
        fragments alone cannot compose — an exact past solution rarely
        solves a new family, whereas a recurring PAIR shortens the
        effective search depth of every program built from it."""
        out = []
        for start in range(len(program)):
            for stop in range(start + 2, len(program) + 1):
                out.append(program[start:stop])
        return out

    def pays_for_itself(fragment: tuple, solved: list) -> bool:
        """Minimum description length, in instructions. Naming a
        fragment costs its own length once; each later use spends one
        token instead of `len(fragment)`. Keep it only if the arithmetic
        comes out positive — this is what makes the library a
        COMPRESSION of what has been solved rather than a pile of it."""
        used = sum(occurrences(fragment, program) for program in solved)
        return used * (len(fragment) - 1) - len(fragment) > 0

    # ops that actually read each argument field, so the marginals are
    # not polluted by slots a NOOP never touched
    USES_I = tuple(o != "NOOP" for o in OPS)
    USES_J = tuple(o in ("CINC", "CDEC", "COPY", "SWAP") for o in OPS)

    class Proposer:
        """A proposal distribution over programs, fitted to what has
        actually solved past families.

        Three statistics, all over the AMODAL instruction basis and so
        domain-general by the same argument as the basis itself:

          op marginal    which operations pay off at all
          slot marginals which slots solved programs actually touch
          SKETCHES       op-sequences that recur, arguments left FREE

        The sketch is the piece F157 was missing, and the reason the
        first attempt at this fix added ZERO fragments across nine
        families. Held as a CONCRETE instruction sequence, a fragment
        can essentially never recur: there are NOPS*SLOTS*(SLOTS-1)
        distinct instructions, so two winning programs sharing an exact
        three-instruction run is a coincidence we should never expect to
        see. Held as a SKETCH, "increment then compare" recurs even when
        the slots differ, and the arguments come from the marginals.

        Untrained, this is EXACTLY uniform over the concrete atoms —
        uniform op times uniform slots is the frozen arm's distribution.
        The arms therefore start from the same proposal and differ only
        in what they learn from what worked."""

        def __init__(self, use_sketches: bool):
            self.use_sketches = use_sketches
            self.parts = [(op,) for op in range(NOPS)]   # length-1 = ops
            self.weight = [1.0] * NOPS
            self.exact: list = []          # whole past winning programs
            self.exact_weight: list = []
            self.i_count = [1.0] * SLOTS
            self.j_count = [1.0] * SLOTS
            self.known = {(op,) for op in range(NOPS)}

        def size(self) -> int:
            return len(self.parts) + len(self.exact)

        def tables(self):
            return (torch.tensor(self.weight + self.exact_weight),
                    torch.tensor(self.i_count), torch.tensor(self.j_count),
                    torch.tensor([float(len(p)) for p in self.parts]
                                 + [float(len(p)) for p in self.exact]))

        def draw(self, tables, generator) -> list:
            """One program of args.program_len instructions."""
            element, islot, jslot, widths = tables
            out: list = []
            while len(out) < args.program_len:
                # only draw among elements that FIT the space left.
                # Without this a stored program is chopped whenever
                # something was drawn before it, so exact reuse could
                # never pay off and the arm would look null for a
                # bookkeeping reason rather than a real one. The
                # length-1 ops always fit, so this can never empty.
                room = widths <= (args.program_len - len(out))
                pick = int(torch.multinomial(element * room, 1,
                                             generator=generator))
                if pick >= len(self.parts):
                    out = out + list(self.exact[pick - len(self.parts)])
                    continue
                for op in self.parts[pick]:
                    i = int(torch.multinomial(islot, 1,
                                              generator=generator))
                    masked = jslot.clone()
                    masked[i] = 0.0        # the basis forbids i == j
                    j = int(torch.multinomial(masked, 1,
                                              generator=generator))
                    out.append((op, i, j))
            return out[:args.program_len]

        def observe(self, winners: list, solved: list) -> None:
            """Fit the statistics to programs that WORKED."""
            for program in winners:
                for op, i, j in program:
                    if USES_I[op]:
                        self.i_count[i] += 1.0
                    if USES_J[op]:
                        self.j_count[j] += 1.0
            shapes = [tuple(op for op, _, _ in p) for p in solved]
            fresh = [tuple(op for op, _, _ in p) for p in winners]
            for slot, part in enumerate(self.parts):
                self.weight[slot] += sum(occurrences(part, shape)
                                         for shape in fresh)
            if not self.use_sketches:
                return
            for program in winners:
                self.exact.append(tuple(program))
                self.exact_weight.append(1.0)
            for shape in fresh:
                for part in sub_fragments(shape):
                    if part in self.known:
                        continue
                    if not pays_for_itself(part, shapes):
                        continue
                    self.known.add(part)
                    self.parts.append(part)
                    self.weight.append(float(sum(
                        occurrences(part, s) for s in shapes)))

    def search_with_proposer(family, proposer, generator, budget):
        """Identical to search_with_library except for where candidates
        come from. Cost is still counted in plant forward passes, which
        is what makes the arms comparable."""
        states, acts, nexts = observe(family, args.observations,
                                      generator)
        used = (states < VALUES).all(dim=0)
        states = torch.where(states < VALUES, states,
                             torch.zeros_like(states))
        nexts = torch.where(nexts < VALUES, nexts,
                            torch.zeros_like(nexts))
        tables = proposer.tables()
        recipe, tried_total, fits = {}, 0, []
        for action in range(family.actions):
            keep = acts == action
            if int(keep.sum()) < 4:
                continue
            src, dst = states[keep], nexts[keep]
            best, best_score, tried = None, -1.0, 0
            while tried < budget:
                candidate = proposer.draw(tables, generator)
                tried += 1
                with torch.no_grad():
                    got = plant(candidate, src).argmax(-1)
                score = float((got[:, used] == dst[:, used])
                              .float().mean())
                if score > best_score:
                    best, best_score = candidate, score
                if best_score >= args.fit_target:
                    break
            recipe[action] = best
            fits.append(best_score)
            tried_total += tried
        return recipe, tried_total, sum(fits) / max(len(fits), 1)

    def library_run(mode: str, targets: list, budget: int, seed: int):
        """One pass over the family SEQUENCE under one library policy.

        Four policies, forming an attribution ladder:
          frozen    concrete atoms, uniform            (F157 control)
          uniform   + whole recipes, uniform           (F157 growth arm)
          marginal  op and slot marginals only, no stored programs
          sketch    + MDL-selected op-sketches + exact past winners

        `marginal` is the control that decides how the result reads: it
        learns WHICH INSTRUCTIONS pay off without storing a single
        program. If `sketch` only matches it, the gain is instruction
        statistics rather than program reuse, and that distinction is
        the entire reason this arm exists."""
        atoms = [tuple([(op, i, j)]) for op in range(NOPS)
                 for i in range(SLOTS) for j in range(SLOTS) if i != j]
        proposer = (Proposer(mode == "sketch")
                    if mode in ("marginal", "sketch") else None)
        library, weights, solved, sequence = atoms, None, [], []
        started = proposer.size() if proposer else len(atoms)
        # per-arm generator seeded identically, so the arms see the SAME
        # observations for the same family and the comparison is paired
        generator = torch.Generator().manual_seed(seed)
        for name, family in targets:
            if proposer is not None:
                recipe, tried, fit = search_with_proposer(
                    family, proposer, generator, budget)
            else:
                recipe, tried, fit = search_with_library(
                    family, library, weights, generator, budget)
            actions = max(1, sum(1 for v in recipe.values()
                                 if v is not None))
            sequence.append({
                "family": name, "candidates_tried": tried,
                "fit": round(fit, 4),
                "library_size": (proposer.size() if proposer
                                 else len(library)),
                "saturated": tried >= budget * actions})
            winners = [tuple(program) for program in recipe.values()
                       if program]
            if mode == "frozen" or not winners:
                continue
            solved.extend(winners)
            if mode == "uniform":
                for program in winners:
                    library.append(program)
                continue
            proposer.observe(winners, solved)
        return {"sequence": sequence, "started": started,
                "ended": proposer.size() if proposer else len(library),
                "total_candidates": sum(s["candidates_tried"]
                                        for s in sequence),
                "sketches": (sorted(
                    ("+".join(OPS[o] for o in p), round(w, 1))
                    for p, w in zip(proposer.parts, proposer.weight)
                    if len(p) > 1) if proposer else [])}

    gen = torch.Generator().manual_seed(args.seed * 104729)
    targets = [("line", Family("line")), ("dial", Family("dial")),
               ("toggle", Family("toggle")), ("perm", Family("perm")),
               ("grid", Family("grid"))]
    for index in range(2):
        targets.append((f"proc{index}",
                        RandomFamily(random_family_spec(gen))))
    if not args.library_arms:
        report["synthesis"] = {name: synthesise(fam, gen)
                               for name, fam in targets}

    def related_specs(generator, count: int) -> list:
        """Families sharing the state GEOMETRY but not the action
        effects: same slot count, value count and space, independently
        drawn ops.

        Reuse can only pay to the extent that solved programs say
        something about unsolved ones. A sequence of families chosen to
        share as little as possible — which is what BREADTH asks for —
        is therefore the condition under which reuse is least able to
        help, and reporting a null there without this contrast would
        confuse 'reuse does not work' with 'these tasks share nothing'.
        Shared structure has to be a knob, not an assumption."""
        base = random_family_spec(generator)
        out, attempts = [base], 0
        while len(out) < count:
            attempts += 1
            if attempts > 20000:
                # the 17-hour lesson: a uniqueness loop that cannot be
                # satisfied must fail loudly, not spin
                raise SystemExit(
                    f"could not draw {count} families matching "
                    f"slots={base['slots']} values={base['values']} "
                    f"space={base['space']} in {attempts} attempts")
            spec = random_family_spec(generator)
            if (spec["slots"] == base["slots"]
                    and spec["values"] == base["values"]
                    and spec["space"] == base["space"]):
                out.append(spec)
        return out

    if args.library_arms:
        # every arm searches against the SAME frozen plant, which
        # removes training variance from the comparison entirely and
        # costs one training run instead of eight
        sequences = {}
        diverse = list(targets)
        for index in range(args.extra_families):
            diverse.append((f"extra{index}",
                            RandomFamily(random_family_spec(gen))))
        sequences["diverse"] = diverse
        if args.related_families:
            sequences["related"] = [
                (f"rel{index}", RandomFamily(spec)) for index, spec
                in enumerate(related_specs(gen, args.related_families))]
        report["library_arms"] = {
            tag: {mode: library_run(mode, seq, args.synthesize,
                                    args.seed * 7907)
                  for mode in ("frozen", "uniform", "marginal", "sketch")}
            for tag, seq in sequences.items()}
        report["library_families"] = {
            tag: [n for n, _ in seq] for tag, seq in sequences.items()}

    if args.library:
        # every single instruction is a fragment to begin with
        library = [[(op, i, j)] for op in range(NOPS)
                   for i in range(SLOTS) for j in range(SLOTS)
                   if i != j]
        started = len(library)
        sequence = []
        for name, family in targets:
            recipe, tried, fit = search_with_library(
                family, library, None, gen, budget=args.synthesize)
            sequence.append({
                "family": name, "candidates_tried": tried,
                "fit": round(fit, 4), "library_size": len(library)})
            if not args.no_growth:
                for program in recipe.values():
                    if program:
                        library.append(list(program))
        report["library_sequence"] = sequence
        report["library_started"] = started
        report["library_ended"] = len(library)

print(json.dumps(report, indent=2))
if args.json:
    with open(args.json, "w") as handle:
        json.dump(report, handle, indent=2)

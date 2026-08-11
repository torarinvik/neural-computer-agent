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
    "--proposal-mass", default="1",
    help="comma-separated sweep. How much weight ONE family's evidence "
         "adds to the proposal distribution, against a prior of 1.0 per "
         "element. 1 is one vote per family. Raw counts (roughly 24 per "
         "family) collapse exploration after a single success — the "
         "first three-seed run had the learned arms spending the whole "
         "4000 budget on families the uniform control solved in 30 "
         "candidates. This knob turns 'how much concentration helps' "
         "into a measurement instead of a guess; 0 recovers uniform.")
parser.add_argument(
    "--exact-weight", default="1,4",
    help="comma-separated sweep of the weight a STORED PROGRAM carries "
         "in the proposal, separately from the marginal mass. The first "
         "corrected seed put this question on the table: on families "
         "sharing their geometry, the arm that reused whole programs "
         "from a uniform pool cost 0.288 of the frozen control, while "
         "the arms that learned instruction statistics cost 1.12-1.21 "
         "— WORSE than frozen. If that replicates, what pays is "
         "recalling a program rather than knowing which instructions "
         "are good, and the sampling probability of stored programs is "
         "the lever. 0 drops stored programs entirely, which isolates "
         "the sketches.")
parser.add_argument(
    "--related-families", type=int, default=0,
    help="also run the whole arm comparison on a sequence of families "
         "that SHARE their state geometry. The diverse sequence is the "
         "condition where reuse is least able to help by construction, "
         "so a null there alone cannot distinguish a broken library from "
         "tasks with nothing in common.")
parser.add_argument(
    "--arms", default="",
    help="comma-separated subset of arm labels to run. Three seeds "
         "found the statistics arms null, so spending their compute on "
         "more SEEDS of the arms that moved is the better trade — a 7 "
         "percent effect needs seeds, not more variants.")
parser.add_argument("--related-seed", type=int, default=20260811,
                    help="draw for the related sequence, held FIXED "
                         "across run seeds so the task set is a "
                         "constant and only search varies")
parser.add_argument("--fit-target", type=float, default=0.95,
                    help="search stops once a candidate reaches this "
                         "fit, so COST (candidates tried) is the "
                         "measurement rather than final accuracy")
parser.add_argument(
    "--moduli", action="store_true",
    help="give every instruction a MODULUS argument, so an instruction "
         "is (op, i, j, m) and INC/DEC wrap at m rather than at a global "
         "VALUES. F160 found this is the real expressibility hole: "
         "`toggle` holds two values per slot, so INC mod 8 is right "
         "exactly half the time, and `perm` needs only swaps and is "
         "therefore immune and scores 1.0000. This widens the "
         "per-instruction space from NOPS*SLOTS*(SLOTS-1) to that times "
         "VALUES-1, so the honest accounting is expressibility gained "
         "against search made harder.")
parser.add_argument(
    "--cover-filter", action="store_true",
    help="reject a candidate before running the interpreter unless every "
         "slot the action CHANGED is written by some instruction in it. "
         "A slot no instruction writes cannot change, so this can never "
         "exclude a program that would have fitted — unlike filtering "
         "per instruction, which is unsound because a correct program "
         "may write a scratch slot and restore it, and which was "
         "measured excluding its own solution (fit 0.887 -> 0.682). The "
         "check costs no forward pass, so it converts search cost from "
         "candidates PROPOSED into candidates EVALUATED, and the report "
         "carries both.")
parser.add_argument(
    "--infer-moduli", action="store_true",
    help="OBSERVE each slot's modulus instead of searching it. Implies "
         "--moduli for the interpreter, which must still be trained on "
         "all of them, but the SEARCH fixes m per slot from the largest "
         "value that slot is ever seen holding. F162 showed the modulus "
         "argument buys expressibility (+0.0439 on short-ranged "
         "families) and costs a seven-times wider search (-0.0108 on "
         "full-range ones). This should keep the first and remove the "
         "second, because the instruction space returns to its original "
         "size once m is determined by i.")
parser.add_argument(
    "--fit-bound", action="store_true",
    help="reject a candidate before running the interpreter when the "
         "mismatches it CANNOT reach already put it below "
         "--fit-target. Sound against the objective the search actually "
         "has, unlike the coverage filter of F163, which was sound for "
         "EXACT fitting and measured a null (1.029, 1.020) because the "
         "search stops at 0.95 and was happy with candidates it "
         "rejected.")
parser.add_argument(
    "--gated-targets", action="store_true",
    help="add `walled` and two procedurally gated families to the "
         "synthesis targets. These are the families an equality guard "
         "would be built for — our CINC/CDEC gate only on 'slot j is "
         "non-zero', so an effect conditional on a slot holding a "
         "PARTICULAR value cannot be written. Measure the failure "
         "before extending the basis; that ordering is why the modulus "
         "was worth adding.")
parser.add_argument(
    "--enum-budget", type=float, default=1.0,
    help="fraction of an action's budget the enumeration may spend "
         "before falling back to sampling. F173's entire loss was a "
         "failed enumeration paid for in full; every WIN it recorded "
         "arrived within 328 calls, so capping should keep the wins and "
         "bound the loss. MEASURED AND REFUTED (F175): capping helps the "
         "diverse sequence 0.425 -> 0.374 and HURTS the related one "
         "0.406 -> 0.503, a net loss. A failed enumeration is not "
         "wasted — it raises the best score the sampling then starts "
         "from — so cutting it saves calls in one place and spends them "
         "in another. Default 1.0, uncapped.")
parser.add_argument("--curve-every", type=int, default=0)
parser.add_argument("--json", default="")
args = parser.parse_args()

# PIN THE THREAD COUNT IN-SCRIPT. Two arms launched with different
# OMP_NUM_THREADS diverge during training at the same seed, because the
# reduction order changes and the arithmetic is not associative. That
# broke a comparison I had described as exactly paired: seeds launched
# with OMP=2 and OMP=1 produced different plants, while the two seeds
# launched together produced identical ones. Pinning here makes the
# pairing a property of the script rather than of how it was invoked.
torch.set_num_threads(1)
torch.manual_seed(args.seed)
SLOTS, VALUES = args.slots, args.values

# The basis. Domain-general by construction: these are operations on
# ABSTRACT slots, identical for a rule family and a grid world.
OPS = ("NOOP", "INC", "DEC", "CINC", "CDEC", "COPY", "SWAP")
if args.no_conditionals:
    OPS = tuple(o for o in OPS if o not in ("CINC", "CDEC"))
NOPS = len(OPS)
NOOP_OP = OPS.index("NOOP")


# Legal moduli an instruction may carry. F160: the instructions did
# arithmetic mod VALUES while every family carries its OWN value count,
# so INC on a slot holding 1 in a two-valued family gave 2 instead of
# 0 — right when the value was 0 and wrong when it was 1, verified at
# exactly 50% on toggle. The modulus is observable in the data and
# names no domain, so making it an ARGUMENT is the domain-general fix:
# an instruction becomes (op, i, j, m) and the interpreter learns
# modular arithmetic parameterised by m exactly as it already learns
# which slot to touch.
MODULI = (tuple(range(2, VALUES + 1))
          if (args.moduli or args.infer_moduli) else (VALUES,))
NMOD = len(MODULI)


def run_instruction(state: torch.Tensor, op: int, i: int,
                    j: int, m: int = 0) -> torch.Tensor:
    """Ground truth. `state` is (batch, SLOTS) of integers < VALUES."""
    name = OPS[op]
    modulus = MODULI[m]
    out = state.clone()
    if name == "NOOP":
        return out
    if name == "INC":
        out[:, i] = (state[:, i] + 1) % modulus
    elif name == "DEC":
        out[:, i] = (state[:, i] - 1) % modulus
    elif name == "CINC":
        gate = state[:, j] != 0
        out[:, i] = torch.where(gate, (state[:, i] + 1) % modulus,
                                state[:, i])
    elif name == "CDEC":
        gate = state[:, j] != 0
        out[:, i] = torch.where(gate, (state[:, i] - 1) % modulus,
                                state[:, i])
    elif name == "COPY":
        out[:, i] = state[:, j]
    elif name == "SWAP":
        out[:, i], out[:, j] = state[:, j], state[:, i]
    return out


def run_program(program: list, state: torch.Tensor) -> torch.Tensor:
    for op, i, j, m in program:
        state = run_instruction(state, op, i, j, m)
    return state


def random_program(generator: torch.Generator, length: int,
                   mod_of_slot: list | None = None) -> list:
    """`mod_of_slot` fixes each slot's modulus instead of searching it.

    F162 measured the modulus argument buying +0.0439 fit on families
    whose value range is narrower than VALUES and costing -0.0108 on
    families at full range, with zero crossovers in fourteen — the
    expressibility is real and the price is a seven-times wider search
    that most families cannot use. But the modulus never had to be
    SEARCHED: each slot's value range is visible in the transitions.
    Reading it off the data names no domain, so this keeps the
    expressibility and returns the instruction space to its original
    size."""
    out = []
    for _ in range(length):
        op = int(torch.randint(0, NOPS, (1,), generator=generator))
        i = int(torch.randint(0, SLOTS, (1,), generator=generator))
        j = int(torch.randint(0, SLOTS, (1,), generator=generator))
        if i == j:
            j = (j + 1) % SLOTS
        if mod_of_slot is not None:
            m = mod_of_slot[i]
        else:
            m = int(torch.randint(0, NMOD, (1,), generator=generator))
        out.append((op, i, j, m))
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
        self.arg_m = torch.nn.Embedding(NMOD, dim)
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
        for op, i, j, m in program:
            code = (self.op(torch.tensor(op))
                    + self.arg_i(torch.tensor(i))
                    + self.arg_j(torch.tensor(j))
                    + self.arg_m(torch.tensor(m)))
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
    "threads": torch.get_num_threads(),
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

    # Which slots can an instruction possibly change? A static property
    # of the INSTRUCTION SET, not of any task, so consulting it is free
    # and carries no knowledge of any domain.
    def written_slots(instruction) -> set:
        op, i, j, _ = instruction
        name = OPS[op]
        if name == "NOOP":
            return set()
        if name == "SWAP":
            return {i, j}
        return {i}

    def writes_needed(src: torch.Tensor, dst: torch.Tensor) -> set:
        """Slots this action actually changed in the observations."""
        return set((src != dst).any(dim=0).nonzero().flatten().tolist())

    def covers(candidate, needs: set) -> bool:   # noqa: C901
        """A SOUND necessary condition: every slot that changed must be
        written by SOME instruction in the program.

        This is the honest form of effect indexing here. The first
        version filtered per instruction — only propose instructions
        that write a changed slot — and that is UNSOUND, because a
        correct program may write a scratch slot and restore it. It
        showed up immediately as a family whose fit fell from 0.887 to
        0.682 when the arm excluded its own solution.

        Requiring COVERAGE cannot exclude anything: a slot that no
        instruction writes cannot change, so a candidate failing this
        test provably cannot fit. It is a rejection rather than a
        preference, and it is checked WITHOUT running the interpreter,
        which is where the cost of search actually lives."""
        if not args.cover_filter or not needs:
            return True
        seen: set = set()
        for instruction in candidate:
            seen |= written_slots(instruction)
        return needs <= seen

    def infer_moduli(*tensors) -> list:
        """Each slot's modulus, read straight off the observations.

        The largest value a slot is ever seen holding fixes its range,
        so `m_i = max observed + 1`. Clamped into the legal set. This is
        a statistic of the data, not knowledge of any task: the same
        line runs for a dial-turning rule family and a foraging grid."""
        stack = torch.cat([t for t in tensors], dim=0)
        out = []
        for slot in range(SLOTS):
            column = stack[:, slot]
            column = column[column < VALUES]
            wanted = (int(column.max()) + 1) if column.numel() else VALUES
            wanted = min(max(wanted, MODULI[0]), MODULI[-1])
            # nearest legal modulus at or above the observed range
            out.append(min(k for k, m in enumerate(MODULI) if m >= wanted))
        return out

    def enumerate_programs(needs: set, moduli, depth: int):
        """Programs of length `depth`, over instructions that WRITE a
        slot the action changed, padded with NOOP to program_len.

        This is the piece the search has never had. Everything measured
        so far SAMPLES: F155 drew random length-6 programs, and every
        improvement since — reuse, the coverage filter, the modulus —
        changes which samples get drawn or which get discarded, so each
        shaves a constant factor off an exponential. Enumeration in
        order of increasing length changes what is being counted: a
        family whose recipe is one instruction costs the size of the
        instruction set, not a fraction of 210^6.

        Restricting to instructions that write a changed slot is the
        same sound condition as the coverage filter — a slot nothing
        writes cannot change — so nothing expressible at this depth is
        skipped. NOOP is excluded from the enumeration itself and used
        only as padding, since a NOOP inside a length-k program makes it
        a length-(k-1) program already enumerated.

        The modulus comes from the slot rather than the enumeration,
        which is F167's result carried across: it is the difference
        between 210 candidates per depth and 1470."""
        pad = [(NOOP_OP, 0, 1, 0)] * args.program_len
        pool = []
        for op in range(NOPS):
            if op == NOOP_OP:
                continue
            targets = needs if needs else range(SLOTS)
            for i in targets:
                for j in range(SLOTS):
                    if i == j:
                        continue
                    pool.append((op, i, j, moduli[i] if moduli else 0))
        if depth == 1:
            for one in pool:
                yield [one] + pad[:args.program_len - 1]
            return
        for first in pool:
            for second in pool:
                yield [first, second] + pad[:args.program_len - 2]

    def attainable(candidate, unavoidable, rows: int, slots: int,
                   used_slots) -> float:
        """The BEST fit this candidate could possibly reach.

        A slot the candidate never writes keeps its input value, so
        every row where that slot changed is a mismatch no execution can
        avoid. Summing those gives an upper bound on fit, and rejecting
        a candidate whose bound is already below `--fit-target` is sound
        AGAINST THE OBJECTIVE SEARCH ACTUALLY HAS.

        This is F163's filter repaired. That one required every changed
        slot to be written, which is sound for EXACT fitting and wrong
        here: a slot that changes in 2 of 64 rows costs 2 mismatches out
        of hundreds, and the search stopping at 0.95 was happy to accept
        it. Measured as a null, 1.029 and 1.020. The bound below permits
        exactly those candidates and rejects only the ones whose
        unavoidable error already exceeds the budget.

        One caveat, stated rather than assumed: the bound is exact for
        GROUND-TRUTH semantics while the search scores with the plant.
        A candidate the bound rejects could in principle score above
        target through interpreter error — but that would be selecting a
        program the interpreter mispredicts, which is not a recipe worth
        keeping."""
        written: set = set()
        for instruction in candidate:
            written |= written_slots(instruction)
        missed = sum(unavoidable[s] for s in used_slots
                     if s not in written)
        return 1.0 - missed / max(rows * slots, 1)

    def covers_set(candidate, needs: set) -> bool:
        """`covers` without the global flag, so an ARM can turn the
        filter on while the other arms keep the old behaviour and stay
        comparable to everything already measured."""
        seen: set = set()
        for instruction in candidate:
            seen |= written_slots(instruction)
        return needs <= seen

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
        # Infer the modulus ONCE from every observation, not per action.
        # Inferring from one action's rows is a smaller sample, and
        # UNDER-estimating a slot's range is unsound in a way
        # over-estimating is not: a modulus below the true value count
        # makes INC wrap early and be wrong, while a modulus above it
        # simply never reaches the wrap. `line` is the tell — it holds
        # eight values and swung +0.1061, -0.0286, -0.0735, 0.0000
        # across four seeds, which is what a modulus that is sometimes
        # too small looks like.
        family_moduli = (infer_moduli(states, nexts)
                         if args.infer_moduli else None)
        recipe, fits = {}, []
        proposed_total = 0
        for action in range(family.actions):
            keep = acts == action
            if int(keep.sum()) < 4:
                continue
            src, dst = states[keep], nexts[keep]
            needs = writes_needed(src, dst)
            fixed = family_moduli
            # per-slot mismatch mass, computed ONCE per action
            live = [k for k in range(SLOTS) if bool(used[k])]
            unavoidable = {k: int((src[:, k] != dst[:, k]).sum())
                           for k in live}
            rows = int(src.shape[0])
            best, best_score, evaluated = None, -1.0, 0
            while evaluated < args.synthesize:
                candidate = random_program(generator, args.program_len,
                                           fixed)
                proposed_total += 1
                if not covers(candidate, needs):
                    continue
                if args.fit_bound and attainable(
                        candidate, unavoidable, rows, len(live),
                        live) < args.fit_target:
                    continue
                evaluated += 1
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
                "identity": round(identity / max(total, 1), 4),
                "proposed": proposed_total,
                "evaluated": args.synthesize * len(fits),
                "kept_fraction": round(
                    args.synthesize * len(fits) / max(proposed_total, 1), 4)}

    def search_with_library(family, library, weights, observer, generator,
                            budget, effect_index=False, cover=False,
                            bound=False, enumerate_first=False,
                            moduli=None):
        """Propose programs by concatenating LIBRARY FRAGMENTS. Returns
        the recipe and the number of candidates tried — cost is the
        measurement here, not accuracy.

        `weights` is None for uniform proposal (F157's behaviour) or a
        per-fragment usefulness count. Weighted proposal is the whole
        point of the fix: F157 appended fragments to a UNIFORMLY sampled
        pool, so every added fragment made every other fragment RARER,
        and the library got bigger without getting better."""
        states, acts, nexts = observe(family, args.observations,
                                      observer)
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
        proposed_total_lib: list = []
        enum_hits: list = []
        for action in range(family.actions):
            keep = acts == action
            if int(keep.sum()) < 4:
                continue
            src, dst = states[keep], nexts[keep]
            pool = library
            # `table` is sized to `library`, so weighting and effect
            # filtering cannot both index the pool. Only the unweighted
            # path filters.
            if effect_index and table is None:
                # EFFECT INDEXING. Only propose instructions that WRITE
                # to a slot this action actually changed. The constraint
                # comes from the observations themselves, so it carries
                # no knowledge of any domain — it is the same move a
                # planner makes when it retrieves only the rules capable
                # of producing an unmet goal cell.
                #
                # It is a HEURISTIC, not a theorem: a correct program
                # may write a scratch slot and restore it, and those
                # solutions are excluded here. That is exactly why this
                # is an arm rather than a default.
                writes = set((src != dst).any(dim=0).nonzero()
                             .flatten().tolist())
                pool = [f for f in library
                        if all(i in writes
                               for _, i, _, _ in f)] or library
            needs = writes_needed(src, dst) if cover else set()
            live = [k for k in range(SLOTS) if bool(used[k])]
            unavoidable = {k: int((src[:, k] != dst[:, k]).sum())
                           for k in live}
            rows = int(src.shape[0])
            best, best_score, tried, proposed = None, -1.0, 0, 0
            if enumerate_first:
                # depth-ordered enumeration BEFORE any sampling. Costs
                # at most the enumerated set and stops the moment a
                # candidate clears the target, so a family with a short
                # recipe pays the size of the instruction set instead of
                # a fraction of an exponential.
                # CAP the enumeration rather than exhausting it. The
                # whole loss in F173 was "paid for a failed enumeration,
                # then sampled anyway", and interleaving one-for-one
                # would bound that at 2x while HALVING every win.
                # Capping is strictly better here because the wins are
                # all early: the most expensive successful enumeration
                # observed was toggle at 328 calls, and depth 1 is only
                # about 60, so a quarter of a 4000 budget keeps every
                # win on record and bounds the loss at 1.25x instead of
                # the 2.2x measured.
                ceiling = max(1, int(budget * args.enum_budget))
                enum_start = tried
                stop = False
                for depth in (1, 2):
                    if stop:
                        break
                    for cand in enumerate_programs(
                            writes_needed(src, dst), moduli, depth):
                        if tried >= ceiling:
                            stop = True
                            break
                        proposed += 1
                        tried += 1
                        with torch.no_grad():
                            got = plant(cand, src).argmax(-1)
                        sc = float((got[:, used] == dst[:, used])
                                   .float().mean())
                        if sc > best_score:
                            best, best_score = cand, sc
                        if best_score >= args.fit_target or tried >= budget:
                            # WHERE a successful enumeration terminates.
                            # F175 could not settle why capping hurt the
                            # related sequence because this was never
                            # recorded: the guess was that those recipes
                            # are found BETWEEN 400 and 930, so a cap
                            # converts wins into fallbacks. A guess about
                            # a distribution is settled by recording the
                            # distribution.
                            if best_score >= args.fit_target:
                                enum_hits.append(tried - enum_start)
                            stop = True
                            break
            while tried < budget and best_score < args.fit_target:
                # build a candidate by concatenating fragments until it
                # is long enough; a fragment may be a whole past recipe
                candidate: list = []
                while len(candidate) < args.program_len:
                    if table is None:
                        slot = int(torch.randint(
                            0, len(pool), (1,), generator=generator))
                    else:
                        slot = int(torch.multinomial(
                            table, 1, generator=generator))
                    candidate = candidate + list(pool[slot])
                candidate = candidate[:args.program_len]
                proposed += 1
                # SOUND rejection, no forward pass. Cost is counted in
                # interpreter evaluations because that is what actually
                # costs; proposals are arithmetic.
                if needs and not covers_set(candidate, needs):
                    if proposed > budget * 50:
                        break          # pathological filter, do not spin
                    continue
                if bound and attainable(candidate, unavoidable, rows,
                                        len(live), live) < args.fit_target:
                    if proposed > budget * 50:
                        break
                    continue
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
            proposed_total_lib.append(proposed)
        return (recipe, tried_total, sum(fits) / max(len(fits), 1),
                sum(proposed_total_lib), enum_hits)

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

        def __init__(self, use_sketches: bool, mass: float,
                     exact_mass: float):
            self.use_sketches = use_sketches
            self.mass = mass
            # weight given to a STORED PROGRAM, separate from the
            # marginal mass. The first corrected run made this the
            # question: the arm that reused whole programs from a
            # uniform pool beat frozen 3.5x on related families, while
            # the arms that learned statistics were WORSE than frozen.
            # If that holds, what pays is recalling a program, not
            # knowing which instructions are good — and then the
            # sampling probability of stored programs is the lever.
            self.exact_mass = exact_mass
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
                    # uniform, so an untrained proposer stays exactly
                    # uniform over the concrete atoms
                    m = int(torch.randint(0, NMOD, (1,),
                                          generator=generator))
                    out.append((op, i, j, m))
            return out[:args.program_len]

        def observe(self, winners: list, solved: list) -> None:
            """Fit the statistics to programs that WORKED.

            ONE VOTE PER FAMILY. Each family's evidence is normalised to
            total mass `args.proposal_mass` before it is added, against a
            prior of 1.0 on every element. Raw counts were the first
            version and they collapse: a single family contributes ~24
            instructions, so after ONE family a used operation outweighs
            an unused one five to ten times over, and the proposer stops
            being able to find solutions that lie outside its own first
            success. That is not a subtle risk — it is what the first
            three-seed run showed, with families the uniform control
            solved in 30 candidates costing the learned arms the entire
            4000 budget."""
            def blend(counts: list, histogram: dict) -> None:
                total = sum(histogram.values())
                if total <= 0:
                    return
                for key, value in histogram.items():
                    counts[key] += self.mass * value / total

            islot: dict = {}
            jslot: dict = {}
            for program in winners:
                for op, i, j, _ in program:
                    if USES_I[op]:
                        islot[i] = islot.get(i, 0) + 1
                    if USES_J[op]:
                        jslot[j] = jslot.get(j, 0) + 1
            blend(self.i_count, islot)
            blend(self.j_count, jslot)
            shapes = [tuple(op for op, _, _ in p) for p in solved]
            fresh = [tuple(op for op, _, _ in p) for p in winners]
            element = {slot: sum(occurrences(part, shape)
                                 for shape in fresh)
                       for slot, part in enumerate(self.parts)}
            blend(self.weight, {k: v for k, v in element.items() if v})
            if not self.use_sketches:
                return
            for program in (winners if self.exact_mass > 0 else []):
                self.exact.append(tuple(program))
                self.exact_weight.append(self.exact_mass
                                         / max(len(winners), 1))
            for shape in fresh:
                for part in sub_fragments(shape):
                    if part in self.known:
                        continue
                    if not pays_for_itself(part, shapes):
                        continue
                    self.known.add(part)
                    self.parts.append(part)
                    # a new sketch enters at the prior, so it is drawn
                    # about as often as one primitive rather than
                    # dominating on the strength of one appearance
                    self.weight.append(1.0)

    def search_with_proposer(family, proposer, observer, generator, budget):
        """Identical to search_with_library except for where candidates
        come from. Cost is still counted in plant forward passes, which
        is what makes the arms comparable."""
        states, acts, nexts = observe(family, args.observations,
                                      observer)
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
        atoms = [tuple([(op, i, j, m)]) for op in range(NOPS)
                 for i in range(SLOTS) for j in range(SLOTS) if i != j
                 for m in range(NMOD)]
        kind, mass, exact_mass = mode
        proposer = (Proposer(kind == "sketch", mass, exact_mass)
                    if kind in ("marginal", "sketch") else None)
        library, weights, solved, sequence = atoms, None, [], []
        proposed_seq: list = []
        stored: set = set()      # every program any earlier family won
        started = proposer.size() if proposer else len(atoms)
        # Two generators, and they must be SEPARATE. One generator for
        # both roles looked paired and was not: the search consumes a
        # different number of draws in every arm, so from the second
        # family onward each arm was solving a DIFFERENT observation
        # sample and the per-family costs were not comparable at all.
        # Seeding the observer per family index makes every arm face the
        # identical search problem, which is the only thing that makes a
        # per-family cost ratio mean anything.
        generator = torch.Generator().manual_seed(seed)
        for index, (name, family) in enumerate(targets):
            observer = torch.Generator().manual_seed(seed + 7919 * index)
            if proposer is not None:
                recipe, tried, fit = search_with_proposer(
                    family, proposer, observer, generator, budget)
                proposals, hits = tried, []
            else:
                recipe, tried, fit, proposals, hits = search_with_library(
                    family, library, weights, observer, generator, budget,
                    effect_index=(kind == "effect"),
                    cover=kind.startswith("cover"),
                    bound=kind.startswith("bound"),
                    enumerate_first=kind.startswith("enum"),
                    moduli=(infer_moduli(*observe(
                        family, args.observations,
                        torch.Generator().manual_seed(seed + 7919 * index))[
                            0::2]) if args.infer_moduli else None))
            proposed_seq.append(proposals)
            actions = max(1, sum(1 for v in recipe.values()
                                 if v is not None))
            winners = [tuple(program) for program in recipe.values()
                       if program]
            # OBSERVE THE MECHANISM, do not infer it from cost. If reuse
            # is what makes a late family cheap, the winning program
            # should literally BE one an earlier family produced. Cost
            # alone cannot distinguish that from a lucky draw, and this
            # can: it is the difference between "reuse happened" and
            # "the number went down".
            recalled = sum(1 for program in winners if program in stored)
            sequence.append({
                "family": name, "candidates_tried": tried,
                "fit": round(fit, 4),
                "library_size": (proposer.size() if proposer
                                 else len(library)),
                "winners": len(winners), "recalled": recalled,
                "enum_hits": hits,
                "saturated": tried >= budget * actions})
            stored.update(winners)
            if kind in ("frozen", "effect", "cover", "bound", "enum") \
                    or not winners:
                continue
            solved.extend(winners)
            if kind in ("uniform", "cover+store", "bound+store",
                        "enum+store"):
                for program in winners:
                    library.append(program)
                continue
            if kind == "shuffled":
                # THE CAUSAL NULL for storing programs. A stored winner
                # is a full-length element, and drawing one fills the
                # whole program in a single pick — which changes the
                # proposal distribution whatever the element CONTAINS.
                # So append the same number of full-length elements
                # drawn at RANDOM. If this matches the arm that stores
                # real winners, the gain is a sampling artefact of
                # element length and not reuse at all.
                for _ in winners:
                    library.append(tuple(random_program(
                        generator, args.program_len)))
                continue
            proposer.observe(winners, solved)
        return {"sequence": sequence, "started": started,
                "ended": proposer.size() if proposer else len(library),
                "total_candidates": sum(s["candidates_tried"]
                                        for s in sequence),
                "proposed_total": sum(proposed_seq),
                "recalled_total": sum(s["recalled"] for s in sequence),
                "winners_total": sum(s["winners"] for s in sequence),
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
    if args.gated_targets:
        # The families an EQUALITY GUARD would exist for, added BEFORE
        # any guard is built, because the rule that has served this
        # project is to establish the failure signature first. Our
        # conditionals gate on "slot j is non-zero" and nothing else,
        # so an effect that fires only when a slot holds a PARTICULAR
        # value has no expression. `walled` is the hand-made case and
        # F92 already flagged it as the reacher's one decisive failure;
        # `gate*` are procedural families drawn with the wall and cond
        # primitives enabled.
        #
        # If these fit as well as the rest, there is no hole and the
        # guard should not be built. That is the point of measuring
        # before extending: the modulus was worth adding because
        # `toggle` failed loudly first.
        targets.append(("walled", Family("walled")))
        for index in range(2):
            targets.append((f"gate{index}", RandomFamily(
                random_family_spec(gen, gated=True))))
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
            # SAME task set in every seed. Drawing these from the run
            # seed made "related" mean a different geometry per seed,
            # and per-family cost swings two orders of magnitude, so
            # the between-seed variance was task variance rather than
            # search variance — which is what made one seed read 0.288
            # and the next 0.892 on the identical arm. Fixing the draw
            # leaves the plant and the search RNG as the only things
            # that differ.
            fixed = torch.Generator().manual_seed(args.related_seed)
            sequences["related"] = [
                (f"rel{index}", RandomFamily(spec)) for index, spec
                in enumerate(related_specs(fixed,
                                           args.related_families))]
        # (kind, marginal mass, stored-program weight). Mass is fixed at
        # one vote per family throughout; the SWEEP is on how much
        # weight a stored program carries, because that is what the
        # first corrected seed made the live question.
        arms = [("frozen", 0.0, 0.0), ("uniform", 0.0, 0.0),
                ("shuffled", 0.0, 0.0), ("effect", 0.0, 0.0),
                ("cover", 0.0, 0.0), ("cover+store", 0.0, 0.0),
                ("bound", 0.0, 0.0), ("bound+store", 0.0, 0.0),
                ("enum", 0.0, 0.0), ("enum+store", 0.0, 0.0),
                ("marginal", 1.0, 0.0), ("sketch", 1.0, 0.0)]
        arms += [("sketch", 1.0, float(w))
                 for w in args.exact_weight.split(",") if float(w) > 0]
        labels = [kind if kind in ("frozen", "uniform", "shuffled",
                                   "effect", "cover", "cover+store",
                                   "bound", "bound+store",
                                   "enum", "enum+store")
                  else f"{kind}-e{exact:g}"
                  for kind, _, exact in arms]
        if args.arms:
            keep = set(args.arms.split(","))
            missing = keep - set(labels)
            if missing:
                raise SystemExit(f"unknown arms {sorted(missing)}; "
                                 f"available: {labels}")
            arms = [a for a, l in zip(arms, labels) if l in keep]
            labels = [l for l in labels if l in keep]
        report["library_arms"] = {
            tag: {label: library_run(arm, seq, args.synthesize,
                                     args.seed * 7907)
                  for label, arm in zip(labels, arms)}
            for tag, seq in sequences.items()}
        report["library_families"] = {
            tag: [n for n, _ in seq] for tag, seq in sequences.items()}

    if args.library:
        # every single instruction is a fragment to begin with
        library = [[(op, i, j, 0)] for op in range(NOPS)
                   for i in range(SLOTS) for j in range(SLOTS)
                   if i != j]
        started = len(library)
        sequence = []
        for name, family in targets:
            recipe, tried, fit, _, _ = search_with_library(
                family, library, None, gen, gen, budget=args.synthesize)
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

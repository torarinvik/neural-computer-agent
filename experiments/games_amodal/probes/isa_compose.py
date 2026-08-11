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
            latent = self.norm(self.step(
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
print(json.dumps(report, indent=2))
if args.json:
    with open(args.json, "w") as handle:
        json.dump(report, handle, indent=2)

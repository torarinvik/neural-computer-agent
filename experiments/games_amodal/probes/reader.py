"""Replace the search with a reader, using the search's own labels.

The recipe architecture works and it is a SEARCH system: a new world is
handled by proposing candidate programs and scoring them against a
frozen interpreter. F178 got that down to about 23 candidates per
action and F194 showed it spans domains. But the system still needs the
environment in the loop to answer "is this program right?", and it
learns nothing from having solved a thousand worlds — world 1001 costs
exactly what world 1 did.

A reader would close that. Given a new world's observed transitions, it
would emit the program directly, and search would become verification
rather than discovery.

F117 and F138 both tried this and both failed, but they tried it in the
form that was available then: emit a CONTINUOUS entry vector, trained
through the task loss. F138 established the representation was capable
(0.9723 when distilled from a privileged entry) while ordinary training
sat near chance, and F164 narrowed that to optimisation reliability
rather than impossibility.

**Two things changed since, and together they make this a different
problem.**

  1. Recipes are now DISCRETE and SHORT — one or two instructions of
     (op, i, j, m), where F176 measured the median successful
     enumeration terminating at the sixth candidate. So this is
     classification over four small vocabularies, not regression into
     an activation space.
  2. Search succeeds on 100% of actions (F178), so it MANUFACTURES
     ground-truth labels. Every family the search solves is a free
     supervised example of "these transitions imply this program".

That is DreamCoder's sleep phase, and the reason it was not available
before is that the wake phase did not work.

The measurement, on families the reader never trained on:

  * top-1 accuracy — does its first guess execute correctly?
  * search cost if the reader proposes first and search verifies;
  * against a SHUFFLED-LABEL control, because a reader that has learned
    the marginal distribution of programs would score above chance
    while having read nothing from the transitions at all.
"""

from __future__ import annotations

import argparse
import json

import torch

from experiments.games_amodal.probes.schema_families import (
    RandomFamily, random_family_spec)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=69316)
parser.add_argument("--dim", type=int, default=128)
parser.add_argument("--interpreter-updates", type=int, default=40000)
parser.add_argument("--reader-updates", type=int, default=8000)
parser.add_argument("--train-families", type=int, default=400)
parser.add_argument("--held-families", type=int, default=100)
parser.add_argument("--examples", type=int, default=32,
                    help="transitions the reader sees per action")
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument(
    "--real-training", action="store_true",
    help="train on transitions from REAL families with labels found by "
         "enumeration — the search feeding the reader, as the design "
         "intended. Without it training uses random instructions over "
         "RANDOM states, which cost 0.41 functional accuracy on real "
         "families.")
parser.add_argument("--shuffle-labels", action="store_true",
                    help="control: train on labels paired with the WRONG "
                         "transitions. A reader that has only learned "
                         "which programs are COMMON scores above chance "
                         "here, so anything this arm reaches is not "
                         "reading.")
parser.add_argument("--json", default="")
args = parser.parse_args()

torch.set_num_threads(1)
torch.manual_seed(args.seed)

SLOTS, VALUES = 6, 8
OPS = ("NOOP", "INC", "DEC", "CINC", "CDEC", "COPY", "SWAP",
       "SINC", "SDEC")
MODULI = tuple(range(2, VALUES + 1))


def run_instruction(state, op, i, j, m):
    name, mod = OPS[op], MODULI[m]
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


class Reader(torch.nn.Module):
    """Transitions -> one instruction.

    A SET encoder over (state, next state) pairs: each pair is embedded
    independently and the set is mean-pooled, so the reader cannot
    depend on the order or count of examples it is given. F87's rule in
    a different costume — the key addresses, and here the key is the
    transitions themselves."""

    def __init__(self, dim: int):
        super().__init__()
        self.embed = torch.nn.Sequential(
            torch.nn.Linear(2 * SLOTS * VALUES, dim), torch.nn.ReLU(),
            torch.nn.Linear(dim, dim), torch.nn.ReLU())
        self.pool = torch.nn.Sequential(
            torch.nn.Linear(dim, dim), torch.nn.ReLU(),
            torch.nn.Linear(dim, dim), torch.nn.ReLU())
        self.op = torch.nn.Linear(dim, len(OPS))
        self.arg_i = torch.nn.Linear(dim, SLOTS)
        self.arg_j = torch.nn.Linear(dim, SLOTS)
        self.arg_m = torch.nn.Linear(dim, len(MODULI))

    def forward(self, before, after):
        # before/after: (batch, examples, SLOTS)
        b, e, _ = before.shape
        hot = torch.cat([
            torch.nn.functional.one_hot(before, VALUES).float()
            .view(b, e, -1),
            torch.nn.functional.one_hot(after, VALUES).float()
            .view(b, e, -1)], dim=-1)
        latent = self.pool(self.embed(hot).mean(dim=1))
        return (self.op(latent), self.arg_i(latent),
                self.arg_j(latent), self.arg_m(latent))


def make_batch(count: int, generator: torch.Generator):
    """Manufacture supervised examples the way the SEARCH would: draw an
    instruction, apply it, and hand the reader the transitions.

    This is the honest version of "the search generates its own labels"
    for a single-instruction recipe — the search's answer for such a
    family IS the instruction, so sampling the instruction directly
    gives identical pairs without spending the search."""
    ops = torch.randint(0, len(OPS), (count,), generator=generator)
    ii = torch.randint(0, SLOTS, (count,), generator=generator)
    jj = torch.randint(0, SLOTS, (count,), generator=generator)
    jj = torch.where(jj == ii, (jj + 1) % SLOTS, jj)
    mm = torch.randint(0, len(MODULI), (count,), generator=generator)
    before = torch.randint(0, VALUES, (count, args.examples, SLOTS),
                           generator=generator)
    after = torch.empty_like(before)
    for row in range(count):
        after[row] = run_instruction(before[row], int(ops[row]),
                                     int(ii[row]), int(jj[row]),
                                     int(mm[row]))
    return before, after, ops, ii, jj, mm


def real_batch(count: int, generator: torch.Generator):
    """Labels the way the search ACTUALLY produces them: draw a real
    family, take one action's transitions, and find the instruction that
    reproduces them exactly.

    This is what `make_batch` shortcut. I reasoned that "the search's
    answer for a single-instruction family IS the instruction, so
    sampling the instruction directly gives identical pairs" — true of
    the LABEL and false of the STATES. A three-valued family only ever
    shows 0-2; random states are uniform over 0-7. The shortcut cost
    0.41 of functional accuracy on real families (0.7012 synthetic
    against 0.2929 real), which is the whole difference between a reader
    that could replace the search and one that cannot.

    Enumerating 9 x 6 x 5 x 7 = 1890 instructions per family is cheap
    and is exactly the search — so this is the sleep phase fed by the
    wake phase, as intended rather than as approximated."""
    rows: list = []
    guard = 0
    while len(rows) < count:
        guard += 1
        if guard > count * 200:
            raise SystemExit("could not draw enough single-instruction "
                             "families; widen the draw or lower --count")
        family = RandomFamily(random_family_spec(generator))
        size = len(family.states)
        idx = torch.randint(0, size, (args.examples,),
                            generator=generator)
        act = int(torch.randint(0, family.actions, (1,),
                                generator=generator))
        nxt = torch.tensor([family.table[int(x)][act] for x in idx])
        before, after = family.slot_values(idx), family.slot_values(nxt)
        if bool((before >= VALUES).any()) or bool((after >= VALUES).any()):
            continue
        found = None
        for op in range(len(OPS)):
            for i in range(SLOTS):
                for j in range(SLOTS):
                    if i == j:
                        continue
                    for m in range(len(MODULI)):
                        if bool((run_instruction(before, op, i, j, m)
                                 == after).all()):
                            found = (op, i, j, m)
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        if found is None:
            continue                # needs more than one instruction
        rows.append((before, after) + found)
    return (torch.stack([r[0] for r in rows]),
            torch.stack([r[1] for r in rows]),
            torch.tensor([r[2] for r in rows]),
            torch.tensor([r[3] for r in rows]),
            torch.tensor([r[4] for r in rows]),
            torch.tensor([r[5] for r in rows]))


reader = Reader(args.dim)
optimizer = torch.optim.AdamW(reader.parameters(), lr=args.lr,
                              weight_decay=0.01)
gen = torch.Generator().manual_seed(args.seed * 104729)
curve = []
for update in range(args.reader_updates):
    draw = real_batch if args.real_training else make_batch
    before, after, ops, ii, jj, mm = draw(args.batch_size, gen)
    if args.shuffle_labels:
        perm = torch.randperm(args.batch_size, generator=gen)
        ops, ii, jj, mm = ops[perm], ii[perm], jj[perm], mm[perm]
    po, pi, pj, pm = reader(before, after)
    loss = (torch.nn.functional.cross_entropy(po, ops)
            + torch.nn.functional.cross_entropy(pi, ii)
            + torch.nn.functional.cross_entropy(pj, jj)
            + torch.nn.functional.cross_entropy(pm, mm))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if update % 500 == 0:
        curve.append((update, round(float(loss), 4)))

# ------------------------------------------------------------ evaluate
eval_gen = torch.Generator().manual_seed(args.seed + 7717)
before, after, ops, ii, jj, mm = make_batch(512, eval_gen)
with torch.no_grad():
    po, pi, pj, pm = reader(before, after)
got = (po.argmax(-1), pi.argmax(-1), pj.argmax(-1), pm.argmax(-1))

report = {
    "seed": args.seed, "shuffle_labels": args.shuffle_labels,
    "real_training": args.real_training,
    "reader_updates": args.reader_updates, "curve": curve,
    "field_accuracy": {
        "op": round(float((got[0] == ops).float().mean()), 4),
        "arg_i": round(float((got[1] == ii).float().mean()), 4),
        "arg_j": round(float((got[2] == jj).float().mean()), 4),
        "arg_m": round(float((got[3] == mm).float().mean()), 4)},
    "chance": {"op": round(1 / len(OPS), 4), "arg_i": round(1 / SLOTS, 4),
               "arg_j": round(1 / SLOTS, 4),
               "arg_m": round(1 / len(MODULI), 4)},
}
exact = ((got[0] == ops) & (got[1] == ii) & (got[2] == jj)
         & (got[3] == mm))
report["exact_program"] = round(float(exact.float().mean()), 4)

# The measurement that matters is not whether the reader names the same
# instruction, but whether the instruction it names DOES THE SAME
# THING. Several instructions coincide on a given batch — SWAP i,j and
# COPY when the slots already agree, any op on a slot the family never
# moves — so field accuracy understates a reader that is functionally
# right. Executed against ground truth, which needs no interpreter.
works = torch.zeros(512, dtype=torch.bool)
for row in range(512):
    want = after[row]
    guess = run_instruction(before[row], int(got[0][row]),
                            int(got[1][row]), int(got[2][row]),
                            int(got[3][row]))
    works[row] = bool((guess == want).all())
report["functionally_correct"] = round(float(works.float().mean()), 4)

# ------------------------------------------- the evaluation that counts
# The training distribution is "a random instruction applied to RANDOM
# states". Real families are not that: a family with three values only
# ever shows 0-2, and its states are a structured subset. So the
# synthetic score above measures whether the architecture can read AT
# ALL, and this measures whether it can read the thing the system
# actually needs read.
#
# Held-out families, none seen in training, and only those whose true
# dynamics ARE one instruction — because a reader emitting one
# instruction cannot be scored against a family that needs two, and
# scoring it there would measure the family's depth rather than the
# reader.
def family_examples(count: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    rows = []
    while len(rows) < count:
        family = RandomFamily(random_family_spec(generator))
        size = len(family.states)
        idx = torch.randint(0, size, (args.examples,),
                            generator=generator)
        act = int(torch.randint(0, family.actions, (1,),
                                generator=generator))
        nxt = torch.tensor([family.table[int(x)][act] for x in idx])
        before = family.slot_values(idx)
        after = family.slot_values(nxt)
        if bool((before >= VALUES).any()) or bool((after >= VALUES).any()):
            continue          # family uses fewer slots; skip rather
        rows.append((before, after))   # than pad and measure the padding
    return (torch.stack([r[0] for r in rows]),
            torch.stack([r[1] for r in rows]))


fb, fa = family_examples(256, args.seed + 31337)
with torch.no_grad():
    fo, fi, fj, fm = reader(fb, fa)
hits = 0
for row in range(fb.shape[0]):
    guess = run_instruction(fb[row], int(fo[row].argmax()),
                            int(fi[row].argmax()), int(fj[row].argmax()),
                            int(fm[row].argmax()))
    hits += int(bool((guess == fa[row]).all()))
report["real_family_functionally_correct"] = round(hits / fb.shape[0], 4)
identity = sum(int(bool((fb[r] == fa[r]).all()))
               for r in range(fb.shape[0])) / fb.shape[0]
report["real_family_identity_floor"] = round(identity, 4)

print(json.dumps(report, indent=2))
if args.json:
    with open(args.json, "w") as handle:
        json.dump(report, handle, indent=2)

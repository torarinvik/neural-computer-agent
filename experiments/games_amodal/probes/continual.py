"""Continual learning, stated as one measurement.

The project's founding thesis is that skills belong in an external bank
rather than in weights, because weights interfere and banks do not.
Every probe so far has argued this indirectly — retention under EWC,
entry-conditioned prediction, transfer matrices. None of them has put
the two architectures side by side on the SAME task sequence and asked
the definitional question:

    after learning families 1..N in order, how well do you still do
    family 1?

That question is finally cheap to ask. F178 made recipe search succeed
on 100% of actions at 22.9 candidates each, so a whole sequence of
families can be solved and re-solved in seconds, and the measurement is
no longer dominated by whether the search happened to work.

Two arms on one family sequence, one amodal interface, same seeds:

  WEIGHTS   a slot model trained by gradient descent on each family in
            turn, carrying nothing between them but its own weights.
            This is the ordinary continual-learning setup and it is
            expected to forget.

  BANK      the frozen instruction interpreter from `isa_compose`, plus
            one searched recipe per family held outside the weights.
            Nothing about the plant changes as families arrive.

The BANK arm's retention is predicted to be EXACTLY flat, and that
prediction is worth stating in the strong form: not "high", not
"better", but bit-identical, because re-evaluating family 1 after
family 9 runs the same frozen weights over the same stored integers.
Any deviation at all is a bug in the harness rather than forgetting,
and that is precisely what makes it a useful control — a claim that
cannot degrade gracefully will fail loudly if the setup is wrong.

The interesting number is therefore NOT the bank arm. It is the size of
the weights arm's decay, which is what the bank arm is worth.
"""

from __future__ import annotations

import argparse
import json

import torch

from experiments.games_amodal.probes.schema_families import (
    ACTIONS, SLOTS, VALUES, Family, RandomFamily, SlotModel,
    random_family_spec)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=69316)
parser.add_argument("--dim", type=int, default=96)
parser.add_argument("--families", type=int, default=9)
parser.add_argument("--per-family-updates", type=int, default=1500,
                    help="gradient budget the WEIGHTS arm gets per "
                         "family as it arrives")
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--eval-rows", type=int, default=512)
parser.add_argument(
    "--replay", type=int, default=0,
    help="rows of EARLIER families mixed into each update for the "
         "weights arm. The honest control: replay is how the field "
         "actually prevents forgetting, and a bank that only beats a "
         "no-replay baseline has beaten a straw man. 0 disables.")
parser.add_argument("--interpreter-updates", type=int, default=40000,
                    help="budget for the frozen interpreter, trained "
                         "ONLY on random programs over random states — "
                         "no family ever touches its weights")
parser.add_argument("--bank-path", default="",
                    help="write the bank out and read it back, then "
                         "re-score every family from the RESTORED "
                         "copy. A skill that is data must survive a "
                         "round trip exactly.")
parser.add_argument("--json", default="")
args = parser.parse_args()

torch.set_num_threads(1)
torch.manual_seed(args.seed)

# Named here so the report can record it. A report that cannot explain
# its own number is a report that cannot be audited: `interpreter_check`
# came back at 0.45 against a published 0.99 and the run had not
# recorded the budget or the instruction count that would locate it.
OPS_NAMES = ("NOOP", "INC", "DEC", "CINC", "CDEC", "COPY", "SWAP",
             "SINC", "SDEC")


def build_sequence() -> list:
    """Deliberately unlike each other: the hand-made families plus
    procedural ones, so nothing carries between them except structure."""
    out = [("line", Family("line")), ("dial", Family("dial")),
           ("toggle", Family("toggle")), ("perm", Family("perm")),
           ("grid", Family("grid"))]
    generator = torch.Generator().manual_seed(args.seed * 7919)
    index = 0
    while len(out) < args.families:
        out.append((f"proc{index}",
                    RandomFamily(random_family_spec(generator))))
        index += 1
    return out[:args.families]


sequence = build_sequence()


def rollout(family, count: int, generator: torch.Generator):
    size = len(family.states)
    state = torch.randint(0, size, (count,), generator=generator)
    action = torch.randint(0, family.actions, (count,),
                           generator=generator)
    nxt = torch.tensor([family.table[int(s)][int(a)]
                        for s, a in zip(state, action)])
    return family.slot_values(state), action, family.slot_values(nxt)


def score(model: SlotModel, family, seed: int) -> float:
    """Held-out next-state accuracy over the slots the family uses."""
    generator = torch.Generator().manual_seed(seed)
    values, action, target = rollout(family, args.eval_rows, generator)
    with torch.no_grad():
        got = model(values, action).argmax(-1)
    used = target != VALUES
    return round(float((got[used] == target[used]).float().mean()), 4)


def identity_floor(family, seed: int) -> float:
    """Copying the input unchanged. The real floor for slot tasks — F63
    established it scores far above uniform, so uniform is not the
    baseline any of this should be read against."""
    generator = torch.Generator().manual_seed(seed)
    values, _, target = rollout(family, args.eval_rows, generator)
    used = target != VALUES
    return round(float((values[used] == target[used]).float().mean()), 4)


# ---------------------------------------------------------------- weights
model = SlotModel(args.dim)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
data_gen = torch.Generator().manual_seed(args.seed * 6700417)
weights_curve: list = []

for stage, (name, family) in enumerate(sequence):
    for _ in range(args.per_family_updates):
        values, action, target = rollout(family, args.batch_size,
                                         data_gen)
        if args.replay and stage:
            # mix in rows from every family seen so far
            earlier = sequence[torch.randint(
                0, stage, (1,), generator=data_gen)][1]
            rv, ra, rt = rollout(earlier, args.replay, data_gen)
            values = torch.cat([values, rv])
            action = torch.cat([action, ra])
            target = torch.cat([target, rt])
        logits = model(values, action)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, VALUES), target.reshape(-1),
            ignore_index=VALUES)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    weights_curve.append({
        "after": name,
        "scores": {n: score(model, f, args.seed + 991)
                   for n, f in sequence[:stage + 1]}})

report = {
    "seed": args.seed,
    "families": [n for n, _ in sequence],
    "per_family_updates": args.per_family_updates,
    "interpreter_updates": args.interpreter_updates,
    "interpreter_ops": len(OPS_NAMES),
    "replay": args.replay,
    "identity_floor": {n: identity_floor(f, args.seed + 991)
                       for n, f in sequence},
    "weights_curve": weights_curve,
}

# retention: how much of what it could do does it still do
first = {n: weights_curve[i]["scores"][n]
         for i, (n, _) in enumerate(sequence)}
final = weights_curve[-1]["scores"]
report["weights_at_learning"] = first
report["weights_at_end"] = final
report["weights_forgetting"] = {
    n: round(first[n] - final[n], 4) for n in first}
report["weights_mean_forgetting"] = round(
    sum(report["weights_forgetting"].values()) / len(first), 4)

# ------------------------------------------------------------------- bank
# The instruction interpreter, mirroring `isa_compose.py`. Duplicated
# rather than imported because that module runs on import; the
# duplication is made safe by checking this copy reproduces its headline
# unseen-program accuracy, which is reported as `interpreter_check`.
OPS = OPS_NAMES                  # saturating pair from F177 included
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


class Interpreter(torch.nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.load = torch.nn.Linear(SLOTS * VALUES, dim)
        self.op = torch.nn.Embedding(len(OPS), dim)
        self.arg_i = torch.nn.Embedding(SLOTS, dim)
        self.arg_j = torch.nn.Embedding(SLOTS, dim)
        self.arg_m = torch.nn.Embedding(len(MODULI), dim)
        self.step = torch.nn.Sequential(
            torch.nn.Linear(2 * dim, 2 * dim), torch.nn.ReLU(),
            torch.nn.Linear(2 * dim, 2 * dim), torch.nn.ReLU(),
            torch.nn.Linear(2 * dim, dim))
        self.norm = torch.nn.LayerNorm(dim)
        self.head = torch.nn.Linear(dim, SLOTS * VALUES)

    def forward(self, program, state):
        hot = torch.nn.functional.one_hot(
            state, VALUES).float().view(state.shape[0], -1)
        latent = self.load(hot)
        for op, i, j, m in program:
            code = (self.op(torch.tensor(op)) + self.arg_i(torch.tensor(i))
                    + self.arg_j(torch.tensor(j))
                    + self.arg_m(torch.tensor(m)))
            latent = self.norm(latent + self.step(torch.cat(
                [latent, code.unsqueeze(0).expand(latent.shape[0], -1)],
                dim=-1)))
        return self.head(latent).view(-1, SLOTS, VALUES)


def random_program(generator, length):
    out = []
    for _ in range(length):
        op = int(torch.randint(0, len(OPS), (1,), generator=generator))
        i = int(torch.randint(0, SLOTS, (1,), generator=generator))
        j = int(torch.randint(0, SLOTS, (1,), generator=generator))
        m = int(torch.randint(0, len(MODULI), (1,), generator=generator))
        out.append((op, i, j % SLOTS if j != i else (j + 1) % SLOTS, m))
    return out


interp = Interpreter()
# AdamW, NOT Adam. With weight_decay=0.01 the two are not
# interchangeable: Adam folds the decay into the gradient so the
# adaptive scaling amplifies it on exactly the parameters with small
# sparse gradients — the instruction embeddings — and the interpreter
# never learns its own instruction set. Measured: 0.38 against 0.83 at
# a fifth of the budget, loss pinned at 1.7 against uniform 2.079.
opt = torch.optim.AdamW(interp.parameters(), lr=1e-3, weight_decay=0.01)
train_gen = torch.Generator().manual_seed(args.seed * 104729)
for _ in range(args.interpreter_updates):
    prog = random_program(train_gen, 6)
    st = torch.randint(0, VALUES, (args.batch_size, SLOTS),
                       generator=train_gen)
    tgt = st.clone()
    for op, i, j, m in prog:
        tgt = run_instruction(tgt, op, i, j, m)
    loss = torch.nn.functional.cross_entropy(
        interp(prog, st).reshape(-1, VALUES), tgt.reshape(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()

# faithfulness check against isa_compose's published number
check_gen = torch.Generator().manual_seed(args.seed + 5551)
hits = rows = 0
for _ in range(64):
    prog = random_program(check_gen, 6)
    st = torch.randint(0, VALUES, (128, SLOTS), generator=check_gen)
    tgt = st.clone()
    for op, i, j, m in prog:
        tgt = run_instruction(tgt, op, i, j, m)
    with torch.no_grad():
        hits += int((interp(prog, st).argmax(-1) == tgt).sum())
    rows += tgt.numel()
report["interpreter_check"] = round(hits / rows, 4)

for parameter in interp.parameters():
    parameter.requires_grad_(False)


def infer_moduli(*tensors):
    stacked = torch.cat(list(tensors), dim=0)
    out = []
    for slot in range(SLOTS):
        column = stacked[:, slot]
        column = column[column < VALUES]
        want = (int(column.max()) + 1) if column.numel() else VALUES
        out.append(min(m for m in MODULI if m >= max(want, MODULI[0])))
    return out


def enumerate_programs(writes, moduli, depth):
    singles = [(op, i, j, MODULI.index(moduli[i]))
               for op in range(len(OPS)) if OPS[op] != "NOOP"
               for i in sorted(writes) for j in range(SLOTS) if j != i]
    pad = (0, 0, 1, 0)
    if depth == 1:
        for a in singles:
            yield [a] + [pad] * 5
    else:
        for a in singles:
            for b in singles:
                yield [a, b] + [pad] * 4


def solve(family, generator) -> dict:
    """One recipe per action, by enumeration against the FROZEN
    interpreter. Nothing here trains."""
    values, action, target = rollout(family, 64, generator)
    used = (values < VALUES).all(dim=0)
    values = torch.where(values < VALUES, values, torch.zeros_like(values))
    target = torch.where(target < VALUES, target, torch.zeros_like(target))
    moduli = infer_moduli(values, target)
    recipe = {}
    for act in range(family.actions):
        keep = action == act
        if int(keep.sum()) < 4:
            continue
        src, dst = values[keep], target[keep]
        writes = set((src != dst).any(dim=0).nonzero().flatten().tolist())
        best, best_score = None, -1.0
        for depth in (1, 2):
            if best_score >= 0.99:
                break
            for cand in enumerate_programs(writes or {0}, moduli, depth):
                with torch.no_grad():
                    got = interp(cand, src).argmax(-1)
                sc = float((got[:, used] == dst[:, used]).float().mean())
                if sc > best_score:
                    best, best_score = cand, sc
                if best_score >= 0.99:
                    break
        recipe[act] = best
    return recipe


def bank_score(family, recipe, seed: int) -> float:
    generator = torch.Generator().manual_seed(seed)
    values, action, target = rollout(family, args.eval_rows, generator)
    used = (values < VALUES).all(dim=0)
    values = torch.where(values < VALUES, values, torch.zeros_like(values))
    target = torch.where(target < VALUES, target, torch.zeros_like(target))
    hits = total = 0
    for act, cand in recipe.items():
        keep = action == act
        if not bool(keep.any()) or cand is None:
            continue
        with torch.no_grad():
            got = interp(cand, values[keep]).argmax(-1)
        hits += int((got[:, used] == target[keep][:, used]).sum())
        total += int(keep.sum()) * int(used.sum())
    return round(hits / max(total, 1), 4)


bank: dict = {}
search_gen = torch.Generator().manual_seed(args.seed * 15485863)
bank_curve = []
for stage, (name, family) in enumerate(sequence):
    bank[name] = solve(family, search_gen)
    bank_curve.append({
        "after": name,
        "scores": {n: bank_score(dict(sequence)[n], bank[n],
                                 args.seed + 991)
                   for n in list(bank)}})
report["bank_curve"] = bank_curve
bank_first = {n: bank_curve[i]["scores"][n]
              for i, (n, _) in enumerate(sequence)}
bank_final = bank_curve[-1]["scores"]
report["bank_at_learning"] = bank_first
report["bank_at_end"] = bank_final
report["bank_forgetting"] = {n: round(bank_first[n] - bank_final[n], 4)
                             for n in bank_first}
report["bank_mean_forgetting"] = round(
    sum(report["bank_forgetting"].values()) / len(bank_first), 4)

# ------------------------------------------------------- persistence
# A bank that exists only inside one process is not a bank. The whole
# claim of this architecture is that a skill is DATA — integers a
# frozen interpreter reads — so it must survive being written out and
# read back, and the round trip must be exact rather than approximate.
#
# This also makes the size claim concrete and checkable rather than
# asserted: a recipe is `actions x program_len` instructions of four
# small integers each, against the thousands of floats an entry-vector
# bank would need for the same world.
if args.bank_path:
    serialised = {name: {str(act): [list(step) for step in program]
                         for act, program in recipe.items()
                         if program is not None}
                  for name, recipe in bank.items()}
    with open(args.bank_path, "w") as handle:
        json.dump(serialised, handle)
    with open(args.bank_path) as handle:
        loaded = json.load(handle)
    restored = {name: {int(act): [tuple(step) for step in program]
                       for act, program in recipe.items()}
                for name, recipe in loaded.items()}
    after = {n: bank_score(dict(sequence)[n], restored[n], args.seed + 991)
             for n in restored}
    report["bank_after_reload"] = after
    report["reload_exact"] = all(
        after[n] == bank_final[n] for n in after)
    report["bank_bytes"] = len(json.dumps(serialised))
    report["bank_instructions"] = sum(
        len(p) for r in serialised.values() for p in r.values())

print(json.dumps(report, indent=2))
if args.json:
    with open(args.json, "w") as handle:
        json.dump(report, handle, indent=2)

"""Compositional math: do bank entries factor into reusable pieces?

Every result so far — schema families, the games, math_twins — reads
ONE monolithic rule per world. The founding objective ("task A makes
novel task B faster") cashes out differently: seen PIECES, unseen
ARRANGEMENT. This probe measures exactly that.

A world defines two hidden functions over Z_M (M prime):

    f(x) = (x + b) mod M          (additive piece, b hidden)
    g(x) = (a * x) mod M          (multiplicative piece, a hidden)

The reader sees labeled example applications and mints an entry. The
plant receives a PROGRAM — a token string like [f, g, f] — plus an
input x, and must output the program applied right-to-left. Training
covers a subset of programs; the headline measurement is HELD-OUT
COMPOSITIONS: programs never seen during training, over pieces the
entry already carries. Generalising to [g,f,g,f] from training on
short programs is systematic reuse — the compounding claim itself.

Controls:
  * withheld entry, stranger entry (fresh world) as always;
  * the SWAP twin: a world where the roles of f and g are exchanged
    (a and b re-drawn is not enough — same a, b, but f multiplies and
    g adds). A reader that truly factors pieces makes the plant
    execute the swapped program exactly wrong — the composition
    analogue of math_twins' twin-accuracy-zero;
  * held-out WORLDS (new a, b) x held-out PROGRAMS: both axes of
    generalisation, jointly and separately.

Splits: worlds split as usual; programs split by exact token string,
with all length-1 and length-2 programs in training (the pieces must
be observable) and a random half of length-3/4 programs held out.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=69316)
parser.add_argument("--modulus", type=int, default=23)
parser.add_argument("--dim", type=int, default=96)
parser.add_argument("--bank-tokens", type=int, default=8)
parser.add_argument("--examples", type=int, default=12,
                    help="example applications shown to the reader; "
                         "each is (piece token, x, piece(x)) — SINGLE "
                         "applications only, so composition is never "
                         "demonstrated to the reader")
parser.add_argument("--train-updates", type=int, default=12000)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--ignorance", type=float, default=0.5)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--worlds", type=int, default=24)
parser.add_argument("--held-worlds", type=int, default=6)
parser.add_argument("--max-len", type=int, default=4)
parser.add_argument(
    "--curriculum", type=float, default=0.0,
    help="ramp the maximum program LENGTH from 1 to --max-len over "
         "this fraction of training. Measured motivation: only 11%% of "
         "updates land on length-1 programs and 67%% on length>=3, so "
         "the model spends almost all its time on the hardest form of "
         "a task it cannot yet do at all. F114 showed reading works "
         "when the plant's job is a SINGLE application; F120 showed "
         "the ignorance objective is toothless while the model is bad. "
         "A curriculum gets reading established on the readable task "
         "first, then extends it — the bootstrapping F120 identified.")
parser.add_argument(
    "--contrastive-aux", type=float, default=0.0,
    help="contrastive term as an AUXILIARY loss during joint training "
         "rather than a frozen pre-training phase. F139 showed a "
         "contrastive code is discriminative but not shaped for the "
         "binder; F140 showed the binder cannot be given capacity to "
         "compensate (a nonlinear binder drops the oracle ceiling from "
         "0.9983 to 0.6196). So the code must be made task-shaped "
         "WHILE it is made discriminative, not before: the task loss "
         "supplies the shape, the contrastive term supplies the "
         "gradient that breaks F106's deadlock, and neither waits for "
         "the other.")
parser.add_argument(
    "--deep-binder", action="store_true",
    help="make the entry->parameters decoder an MLP instead of a "
         "single linear map. F135's oracle entry was itself a LINEAR "
         "projection of one-hot world parameters, so a linear binder "
         "inverted it trivially — that ceiling may have been partly an "
         "artefact of matching encoders. A contrastively-learned code "
         "(F139) identifies the world but in an arbitrary, likely "
         "nonlinear arrangement, and a linear binder cannot decode it. "
         "This changes only the binder, so it is attributable.")
parser.add_argument(
    "--contrastive", type=float, default=0.0,
    help="NON-PRIVILEGED reader pre-training. For this fraction of "
         "updates, train the reader alone so that two entries read "
         "from DIFFERENT observation samples of the SAME world agree, "
         "and entries from different worlds do not (InfoNCE over a "
         "batch of worlds). Then freeze the reader and train the plant "
         "to bind whatever code it settled on. F138 showed the reader "
         "can produce a usable entry when given a consistent target, "
         "and F136 showed task loss through a frozen plant cannot find "
         "one; the property distillation actually supplied was "
         "CONSISTENCY, which needs no privileged parameters — a "
         "learner always knows which observations came from the same "
         "world. Phase order is reversed from --two-phase: reader "
         "first, plant second.")
parser.add_argument(
    "--distill", action="store_true",
    help="in phase 2, train the reader to MATCH the oracle entry "
         "directly (squared error on the entry vector) instead of "
         "through task loss. Diagnostic, not a proposed mechanism: "
         "the oracle entry is built from privileged parameters, so a "
         "reader trained this way is not a solution. What it answers "
         "is whether the reader CAN represent the required entry from "
         "observations at all. F136 showed phase-2 task loss fails to "
         "move it; if distillation also fails, the reader or its "
         "inputs are inadequate, and if it succeeds, the reader is "
         "capable and the missing piece is purely the training "
         "signal. Requires --two-phase.")
parser.add_argument(
    "--two-phase", type=float, default=0.0,
    help="F75-F79's FROZEN PLANT + AMORTISED READING, applied here. "
         "Train the plant on ORACLE entries for this fraction of "
         "updates, then FREEZE it and train only the reader through "
         "it. F135 measured the plant's side solved (0.9983 with "
         "bound oracle entries) and the reader's side dead (own == "
         "stranger to four decimals) — the classic F106 deadlock, "
         "where a bad reader gives the plant no reason to use entries "
         "and an entry-ignoring plant gives the reader no gradient. "
         "Phase 1 breaks it by building a plant that DEMANDS a "
         "well-formed entry; phase 2 then has a fixed target to aim "
         "at. Requires --iterate --bind-params.")
parser.add_argument(
    "--bind-params", action="store_true",
    help="decode the entry ONCE into one explicit parameter vector "
         "per piece token, then step on (latent, bound parameter) with "
         "no further access to the entry. F134 measured that a plant "
         "given the world exactly still fails above depth 1 (0.5587 "
         "per-bit vs 1.0000 at one world) while the entry is "
         "re-attended at every step — so the same parameters are "
         "re-extracted on each application and any extraction error "
         "compounds with depth. An interpreter binds its arguments "
         "once and then runs the loop; this does the same. Requires "
         "--iterate.")
parser.add_argument(
    "--oracle-entry", action="store_true",
    help="ORACLE SUBSTITUTION on the entry (the F110 technique that "
         "settled the games): replace the reader's output with the "
         "world's TRUE hidden parameters, projected to entry shape. "
         "F120/F122 measured reading to be entirely absent at "
         "multi-world scale (stranger entry bit-identical to own) and "
         "F121 fixed composition only where reading was not needed. "
         "This separates the two: if oracle entries make multi-world "
         "composition work, execution and composition are sound and "
         "READING alone is the constraint; if they do not, the "
         "interface still cannot use per-world content at all.")
parser.add_argument(
    "--train-max-len", type=int, default=0,
    help="LENGTH EXTRAPOLATION split: train on every program of length "
         "<= L and hold out every LONGER one. The default split holds "
         "out half of the length-3/4 programs but trains on the other "
         "half, so both lengths are represented in training and a model "
         "could pass by interpolating within a length. This split "
         "cannot be passed that way: a length-4 program is only "
         "answerable by applying a piece one more time than was ever "
         "demonstrated, which is the sharpest statement of the "
         "puzzle-piece claim.")
parser.add_argument(
    "--iterate", action="store_true",
    help="apply the program ONE PIECE AT A TIME through a shared step "
         "function over a recurrent latent, decoding only at the end. "
         "F119 measured the one-shot interface fitting trained programs "
         "at 1.0000 while sitting at chance on unseen ARRANGEMENTS of "
         "the same pieces — it memorises composite functions instead of "
         "composing. This makes composition structural: same blocks, "
         "same parameter count, no intermediate supervision.")
parser.add_argument("--json", default="")
args = parser.parse_args()

torch.manual_seed(args.seed)
M = args.modulus
F, G = 0, 1  # piece tokens


def make_worlds() -> list[dict]:
    generator = torch.Generator().manual_seed(args.seed * 7919)
    out = []
    for index in range(args.worlds):
        b = int(torch.randint(1, M, (1,), generator=generator))
        a = int(torch.randint(2, M, (1,), generator=generator))
        out.append({"name": f"w{index}", "a": a, "b": b})
    return out


def apply_program(world: dict, program: tuple, x: torch.Tensor,
                  swapped: bool = False) -> torch.Tensor:
    result = x.clone()
    for token in reversed(program):
        is_f = (token == F) != swapped
        result = ((result + world["b"]) % M if is_f
                  else (world["a"] * result) % M)
    return result


def all_programs() -> tuple[list, list]:
    """(train programs, held programs). Length 1-2 always trained;
    half of length 3-4 held out."""
    generator = torch.Generator().manual_seed(args.seed * 104729)
    short, long = [], []
    for length in range(1, args.max_len + 1):
        for program in itertools.product((F, G), repeat=length):
            (short if length <= 2 else long).append(program)
    if args.train_max_len:
        short_by_len, long_by_len = [], []
        for length in range(1, args.max_len + 1):
            for program in itertools.product((F, G), repeat=length):
                (short_by_len if length <= args.train_max_len
                 else long_by_len).append(program)
        return short_by_len, long_by_len
    if not long:
        # --max-len 1: nothing to hold out on the PROGRAM axis. Reuse
        # the trained programs so the held-out WORLD axis (the reading
        # test) still reports, rather than dividing by zero.
        return short, short
    order = torch.randperm(len(long), generator=generator).tolist()
    half = len(long) // 2
    held = [long[i] for i in order[:half]]
    trained = short + [long[i] for i in order[half:]]
    return trained, held


class Reader(torch.nn.Module):
    """Example rows (piece token, x, piece(x)) -> entry."""

    def __init__(self, dim: int, tokens: int):
        super().__init__()
        self.value = torch.nn.Embedding(M, dim)
        self.piece = torch.nn.Embedding(2, dim)
        self.row = torch.nn.Linear(3 * dim, dim)
        self.queries = torch.nn.Parameter(torch.randn(tokens, dim) * 0.02)
        self.blocks = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(
                dim, 4, dim_feedforward=2 * dim, batch_first=True,
                dropout=0.0, norm_first=True) for _ in range(2)])
        self.norm = torch.nn.LayerNorm(dim)

    def forward(self, pieces, x, y) -> torch.Tensor:
        rows = self.row(torch.cat(
            [self.piece(pieces), self.value(x), self.value(y)], dim=-1))
        token = torch.cat([self.queries, rows], dim=0).unsqueeze(0)
        for block in self.blocks:
            token = block(token)
        return self.norm(token[0, :self.queries.shape[0]])


class Plant(torch.nn.Module):
    """(program tokens, x, entry) -> output distribution."""

    def __init__(self, dim: int, max_len: int):
        super().__init__()
        self.value = torch.nn.Embedding(M, dim)
        self.piece = torch.nn.Embedding(2, dim)
        self.position = torch.nn.Embedding(max_len + 1, dim)
        self.blocks = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(
                dim, 4, dim_feedforward=2 * dim, batch_first=True,
                dropout=0.0, norm_first=True) for _ in range(3)])
        self.norm = torch.nn.LayerNorm(dim)
        self.head = torch.nn.Linear(dim, M)
        self.binder = (torch.nn.Sequential(
            torch.nn.Linear(dim, 4 * dim), torch.nn.ReLU(),
            torch.nn.Linear(4 * dim, 4 * dim), torch.nn.ReLU(),
            torch.nn.Linear(4 * dim, 2 * dim))
            if args.deep_binder else torch.nn.Linear(dim, 2 * dim))
        self.apply_bound = torch.nn.Sequential(
            torch.nn.Linear(2 * dim, 2 * dim), torch.nn.ReLU(),
            torch.nn.Linear(2 * dim, dim))

    def bind(self, entry) -> torch.Tensor:
        """entry -> (2, dim): one bound parameter vector per piece."""
        return self.binder(entry.mean(dim=0)).view(2, -1)

    def step_bound(self, token: int, hidden, params) -> torch.Tensor:
        """One piece applied using its ALREADY-BOUND parameter. The
        entry is not consulted here — that is the point."""
        return self.apply_bound(torch.cat(
            [hidden, params[token].unsqueeze(0).expand(
                hidden.shape[0], -1)], dim=-1))

    def step(self, token: int, hidden, entry):
        """One piece applied to the latent. Shared across positions and
        program lengths — that sharing IS the compositional prior."""
        row = (hidden + self.piece(torch.tensor(token))).unsqueeze(1)
        if entry is not None:
            context = entry.unsqueeze(0).expand(hidden.shape[0], -1, -1)
            row = torch.cat([context, row], dim=1)
        for block in self.blocks:
            row = block(row)
        return self.norm(row[:, -1])

    def forward(self, program: tuple, x, entry) -> torch.Tensor:
        if args.iterate and args.bind_params and entry is not None:
            params = self.bind(entry)
            hidden = self.value(x)
            for token in reversed(program):
                hidden = self.step_bound(token, hidden, params)
            return self.head(self.norm(hidden))
        if args.iterate:
            hidden = self.value(x)
            for token in reversed(program):
                hidden = self.step(token, hidden, entry)
            return self.head(hidden)
        batch = x.shape[0]
        length = len(program)
        tokens = self.piece(torch.tensor(program)) \
            + self.position(torch.arange(length))
        tokens = tokens.unsqueeze(0).expand(batch, -1, -1)
        query = (self.value(x)
                 + self.position(torch.full_like(x, length))).unsqueeze(1)
        row = torch.cat([tokens, query], dim=1)
        if entry is not None:
            context = entry.unsqueeze(0).expand(batch, -1, -1)
            row = torch.cat([context, row], dim=1)
        for block in self.blocks:
            row = block(row)
        return self.head(self.norm(row[:, -1]))


class OracleEntry(torch.nn.Module):
    """Ground-truth world parameters projected into entry shape. Not a
    learner — a substitution, legitimate for an ablation only."""

    def __init__(self, dim: int, tokens: int, width: int):
        super().__init__()
        self.project = torch.nn.Linear(width, dim * tokens)
        self.tokens, self.dim = tokens, dim

    def forward(self, raw) -> torch.Tensor:
        return self.project(raw).view(self.tokens, self.dim)


reader = Reader(args.dim, args.bank_tokens)
plant = Plant(args.dim, args.max_len)
# One-hot, NOT scalars: a/M and b/M would place all worlds' entries on
# a 2-dimensional manifold and force the plant to carve modular
# arithmetic out of a continuous line — the oracle arm could then fail
# for reasons that have nothing to do with reading, which is the whole
# quantity it is meant to isolate.
ORACLE_WIDTH = 2 * M
oracle = OracleEntry(args.dim, args.bank_tokens, ORACLE_WIDTH)
optimizer = torch.optim.Adam(
    list(reader.parameters()) + list(plant.parameters())
    + list(oracle.parameters()), lr=args.lr)


def oracle_raw(world: dict) -> torch.Tensor:
    raw = torch.zeros(1, 2 * M)
    raw[0, world["a"]] = 1.0
    raw[0, M + world["b"]] = 1.0
    return raw

worlds = make_worlds()
select = torch.Generator().manual_seed(args.seed * 15485863)
held_index = set(torch.randperm(
    len(worlds), generator=select)[:args.held_worlds].tolist())
train_worlds = [w for i, w in enumerate(worlds) if i not in held_index]
held_worlds = [w for i, w in enumerate(worlds) if i in held_index]
train_programs, held_programs = all_programs()


def reader_examples(world: dict, generator: torch.Generator):
    pieces = torch.randint(0, 2, (args.examples,), generator=generator)
    x = torch.randint(0, M, (args.examples,), generator=generator)
    y = torch.where(pieces == F, (x + world["b"]) % M,
                    (world["a"] * x) % M)
    return pieces, x, y


data_gen = torch.Generator().manual_seed(args.seed * 6700417)
uniform = math.log(M)
phase_one = int(args.train_updates * args.two_phase)
contrast_end = int(args.train_updates * args.contrastive)


def contrastive_loss(batch_worlds: list) -> torch.Tensor:
    """InfoNCE: an entry must match a SECOND reading of its own world
    better than any other world's. Uses only world identity, which the
    learner observes directly."""
    anchors = torch.stack([
        reader(*reader_examples(w, data_gen)).flatten()
        for w in batch_worlds])
    others = torch.stack([
        reader(*reader_examples(w, data_gen)).flatten()
        for w in batch_worlds])
    anchors = torch.nn.functional.normalize(anchors, dim=-1)
    others = torch.nn.functional.normalize(others, dim=-1)
    logits = anchors @ others.T / 0.1
    target = torch.arange(len(batch_worlds))
    return torch.nn.functional.cross_entropy(logits, target)
for update in range(args.train_updates):
    if args.two_phase > 0 and update == phase_one:
        # freeze the plant; from here only the reader learns, and it
        # must produce entries the FIXED plant can already bind
        for parameter in plant.parameters():
            parameter.requires_grad_(False)
        optimizer = torch.optim.Adam(reader.parameters(), lr=args.lr)
    if args.contrastive > 0 and update < contrast_end:
        picks = torch.randperm(
            len(train_worlds), generator=data_gen)[:8].tolist()
        loss = contrastive_loss([train_worlds[i] for i in picks])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        continue
    if args.contrastive > 0 and update == contrast_end:
        for parameter in reader.parameters():
            parameter.requires_grad_(False)
        optimizer = torch.optim.Adam(plant.parameters(), lr=args.lr)
    world = train_worlds[int(torch.randint(
        0, len(train_worlds), (1,), generator=data_gen))]
    pool = train_programs
    if args.curriculum > 0:
        ramp = update / max(1.0, args.train_updates * args.curriculum)
        cap = min(args.max_len, 1 + int(ramp * args.max_len))
        pool = [p for p in train_programs if len(p) <= cap] or train_programs
    program = pool[int(torch.randint(
        0, len(pool), (1,), generator=data_gen))]
    use_oracle = args.oracle_entry or (
        args.two_phase > 0 and update < phase_one)
    entry = (oracle(oracle_raw(world)) if use_oracle
             else reader(*reader_examples(world, data_gen)))
    x = torch.randint(0, M, (args.batch_size,), generator=data_gen)
    y = apply_program(world, program, x)
    if args.distill and args.two_phase > 0 and update >= phase_one:
        loss = torch.nn.functional.mse_loss(
            entry, oracle(oracle_raw(world)).detach())
    else:
        loss = torch.nn.functional.cross_entropy(
            plant(program, x, entry), y)
    if args.ignorance > 0 and not (
            args.two_phase > 0 and update >= phase_one):
        blind = plant(program, x,
                      torch.zeros_like(entry)).log_softmax(-1)
        entropy = -(blind.exp() * blind).sum(-1).mean()
        loss = loss + args.ignorance * (uniform - entropy)
    if args.contrastive_aux > 0:
        picks = torch.randperm(
            len(train_worlds), generator=data_gen)[:8].tolist()
        loss = loss + args.contrastive_aux * contrastive_loss(
            [train_worlds[i] for i in picks])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def entry_of(world: dict, offset: int = 0) -> torch.Tensor:
    if args.oracle_entry:
        with torch.no_grad():
            return oracle(oracle_raw(world))
    generator = torch.Generator().manual_seed(
        args.seed * 31 + hash(world["name"]) % 100000 + offset)
    with torch.no_grad():
        return reader(*reader_examples(world, generator))


stranger_gen = torch.Generator().manual_seed(args.seed * 32452843)


def stranger_entry() -> torch.Tensor:
    b = int(torch.randint(1, M, (1,), generator=stranger_gen))
    a = int(torch.randint(2, M, (1,), generator=stranger_gen))
    return entry_of({"name": f"s{a}_{b}", "a": a, "b": b}, offset=9)


def accuracy(world: dict, programs: list, entry,
             swapped: bool = False) -> float:
    generator = torch.Generator().manual_seed(args.seed * 977)
    correct, total = 0, 0
    with torch.no_grad():
        for program in programs:
            x = torch.randint(0, M, (64,), generator=generator)
            y = apply_program(world, program, x, swapped=swapped)
            predictions = plant(program, x, entry).argmax(-1)
            correct += int((predictions == y).sum())
            total += 64
    return round(correct / total, 4)


def score_world(world: dict) -> dict:
    entry = entry_of(world)
    return {
        # both axes of generalisation, separately and jointly
        "trained_programs": accuracy(world, train_programs, entry),
        "held_programs": accuracy(world, held_programs, entry),
        "withheld": accuracy(world, held_programs, None),
        "stranger": accuracy(world, held_programs, stranger_entry()),
        # SWAP control: truth computed with f/g roles exchanged. A
        # plant that factors pieces should track the swap truth at
        # CHANCE with the unswapped entry (it executes the real roles)
        "swap_truth_with_own_entry": accuracy(
            world, held_programs, entry, swapped=True),
    }


report = {
    "seed": args.seed, "modulus": M, "worlds": args.worlds,
    "held_worlds": args.held_worlds,
    "train_programs": len(train_programs),
    "held_programs": len(held_programs),
    "chance": round(1 / M, 4),
    "held_out_worlds": {w["name"]: score_world(w) for w in held_worlds},
    "trained_worlds": {w["name"]: score_world(w)
                       for w in train_worlds[:len(held_worlds)]},
}
print(json.dumps(report, indent=2))
if args.json:
    with open(args.json, "w") as handle:
        json.dump(report, handle, indent=2)

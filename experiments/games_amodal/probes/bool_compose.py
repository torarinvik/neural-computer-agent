"""Boolean composition: the puzzle-piece question at minimum difficulty.

F117 asked whether bank entries factor into reusable pieces, using
x -> a*x+b over Z_23, and returned a clean null — but the diagnostics
said FIT failed before reading did (0.10-0.13 even on trained worlds
with trained programs). Composed modular multiplication is close to the
hardest thing a small model can be asked to represent, so that null
indicts the arithmetic, not the mechanism.

This is the same experiment with pieces that are trivial to represent
individually, so anything that fails is the COMPOSITION or the READING,
never the piece:

    f(x) = x XOR b          (hidden mask b)
    g(x) = rotate_left(x, k) (hidden shift k)

over W-bit vectors. Both pieces are one-step-learnable. They do NOT
commute — rot(x XOR b) != rot(x) XOR b — so program ORDER matters and
held-out programs are genuinely unseen functions, not relabellings.
Prediction is per-bit, so the output head is W binary decisions rather
than a 23-way softmax over a space the model must first learn to
represent.

World diversity is no longer the constraint either: (2^W - 1) * (W - 1)
distinct worlds, 1785 at W=8, against the 15 that mod-16 admitted in
F115.

Everything else is the established recipe: reader sees SINGLE
applications only (composition is never demonstrated), ignorance
objective, worlds and programs both split, and the four controls —
own entry, withheld, stranger, and the swap twin (f/g roles exchanged),
whose truth a correctly-factored entry should MISS.

Reported metrics: exact-match (all W bits right, chance 2^-W) is the
headline; per-bit accuracy (chance 0.5) shows partial credit.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=69316)
parser.add_argument("--width", type=int, default=8, help="bits per state")
parser.add_argument("--dim", type=int, default=96)
parser.add_argument("--bank-tokens", type=int, default=8)
parser.add_argument("--examples", type=int, default=12)
parser.add_argument("--train-updates", type=int, default=12000)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--ignorance", type=float, default=0.5)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--worlds", type=int, default=64)
parser.add_argument("--held-worlds", type=int, default=8)
parser.add_argument("--max-len", type=int, default=4)
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
W = args.width
F, G = 0, 1


def bits_of(values: torch.Tensor) -> torch.Tensor:
    """Integer tensor -> (n, W) float bit matrix, bit 0 = least sig."""
    shifts = torch.arange(W)
    return ((values.unsqueeze(-1) >> shifts) & 1).float()


def make_worlds() -> list[dict]:
    generator = torch.Generator().manual_seed(args.seed * 7919)
    seen, out = set(), []
    while len(out) < args.worlds:
        b = int(torch.randint(1, 1 << W, (1,), generator=generator))
        k = int(torch.randint(1, W, (1,), generator=generator))
        if (b, k) in seen:
            continue
        seen.add((b, k))
        out.append({"name": f"b{b}k{k}", "b": b, "k": k})
    return out


def apply_piece(world: dict, token: int, x: torch.Tensor,
                swapped: bool) -> torch.Tensor:
    is_f = (token == F) != swapped
    if is_f:
        return x ^ world["b"]
    k, mask = world["k"], (1 << W) - 1
    return ((x << k) | (x >> (W - k))) & mask


def apply_program(world: dict, program: tuple, x: torch.Tensor,
                  swapped: bool = False) -> torch.Tensor:
    result = x.clone()
    for token in reversed(program):
        result = apply_piece(world, token, result, swapped)
    return result


def all_programs() -> tuple[list, list]:
    generator = torch.Generator().manual_seed(args.seed * 104729)
    short, long = [], []
    for length in range(1, args.max_len + 1):
        for program in itertools.product((F, G), repeat=length):
            (short if length <= 2 else long).append(program)
    order = torch.randperm(len(long), generator=generator).tolist()
    half = len(long) // 2
    return (short + [long[i] for i in order[half:]],
            [long[i] for i in order[:half]])


class Reader(torch.nn.Module):
    """Rows of (piece token, x bits, piece(x) bits) -> entry."""

    def __init__(self, dim: int, tokens: int):
        super().__init__()
        self.piece = torch.nn.Embedding(2, dim)
        self.row = torch.nn.Linear(2 * W, dim)
        self.queries = torch.nn.Parameter(torch.randn(tokens, dim) * 0.02)
        self.blocks = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(
                dim, 4, dim_feedforward=2 * dim, batch_first=True,
                dropout=0.0, norm_first=True) for _ in range(2)])
        self.norm = torch.nn.LayerNorm(dim)

    def forward(self, pieces, x, y) -> torch.Tensor:
        rows = self.row(torch.cat([bits_of(x), bits_of(y)], dim=-1)) \
            + self.piece(pieces)
        token = torch.cat([self.queries, rows], dim=0).unsqueeze(0)
        for block in self.blocks:
            token = block(token)
        return self.norm(token[0, :self.queries.shape[0]])


class Plant(torch.nn.Module):
    """(program, x, entry) -> W bit logits."""

    def __init__(self, dim: int, max_len: int):
        super().__init__()
        self.value = torch.nn.Linear(W, dim)
        self.piece = torch.nn.Embedding(2, dim)
        self.position = torch.nn.Embedding(max_len + 1, dim)
        self.blocks = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(
                dim, 4, dim_feedforward=2 * dim, batch_first=True,
                dropout=0.0, norm_first=True) for _ in range(3)])
        self.norm = torch.nn.LayerNorm(dim)
        self.head = torch.nn.Linear(dim, W)

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
        if args.iterate:
            hidden = self.value(bits_of(x))
            for token in reversed(program):
                hidden = self.step(token, hidden, entry)
            return self.head(hidden)
        batch, length = x.shape[0], len(program)
        tokens = (self.piece(torch.tensor(program))
                  + self.position(torch.arange(length)))
        tokens = tokens.unsqueeze(0).expand(batch, -1, -1)
        query = (self.value(bits_of(x))
                 + self.position(torch.full((batch,), length))).unsqueeze(1)
        row = torch.cat([tokens, query], dim=1)
        if entry is not None:
            context = entry.unsqueeze(0).expand(batch, -1, -1)
            row = torch.cat([context, row], dim=1)
        for block in self.blocks:
            row = block(row)
        return self.head(self.norm(row[:, -1]))


reader = Reader(args.dim, args.bank_tokens)
plant = Plant(args.dim, args.max_len)
optimizer = torch.optim.Adam(
    list(reader.parameters()) + list(plant.parameters()), lr=args.lr)

worlds = make_worlds()
select = torch.Generator().manual_seed(args.seed * 15485863)
held_index = set(torch.randperm(
    len(worlds), generator=select)[:args.held_worlds].tolist())
train_worlds = [w for i, w in enumerate(worlds) if i not in held_index]
held_worlds = [w for i, w in enumerate(worlds) if i in held_index]
train_programs, held_programs = all_programs()


def reader_examples(world: dict, generator: torch.Generator):
    pieces = torch.randint(0, 2, (args.examples,), generator=generator)
    x = torch.randint(0, 1 << W, (args.examples,), generator=generator)
    y = torch.where(pieces == F,
                    apply_piece(world, F, x, False),
                    apply_piece(world, G, x, False))
    return pieces, x, y


data_gen = torch.Generator().manual_seed(args.seed * 6700417)
for update in range(args.train_updates):
    world = train_worlds[int(torch.randint(
        0, len(train_worlds), (1,), generator=data_gen))]
    program = train_programs[int(torch.randint(
        0, len(train_programs), (1,), generator=data_gen))]
    entry = reader(*reader_examples(world, data_gen))
    x = torch.randint(0, 1 << W, (args.batch_size,), generator=data_gen)
    y = bits_of(apply_program(world, program, x))
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        plant(program, x, entry), y)
    if args.ignorance > 0:
        blind = plant(program, x, torch.zeros_like(entry))
        # per-bit entropy, maximised at log 2 when the entry-free
        # prediction is a coin flip on every bit
        p = torch.sigmoid(blind).clamp(1e-6, 1 - 1e-6)
        entropy = -(p * p.log() + (1 - p) * (1 - p).log()).mean()
        loss = loss + args.ignorance * (math.log(2) - entropy)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def entry_of(world: dict, offset: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(
        args.seed * 31 + hash(world["name"]) % 100000 + offset)
    with torch.no_grad():
        return reader(*reader_examples(world, generator))


stranger_gen = torch.Generator().manual_seed(args.seed * 32452843)


def stranger_entry() -> torch.Tensor:
    b = int(torch.randint(1, 1 << W, (1,), generator=stranger_gen))
    k = int(torch.randint(1, W, (1,), generator=stranger_gen))
    return entry_of({"name": f"s{b}_{k}", "b": b, "k": k}, offset=9)


def accuracy(world: dict, programs: list, entry,
             swapped: bool = False) -> tuple[float, float]:
    generator = torch.Generator().manual_seed(args.seed * 977)
    exact, bits, total = 0, 0.0, 0
    with torch.no_grad():
        for program in programs:
            x = torch.randint(0, 1 << W, (64,), generator=generator)
            y = bits_of(apply_program(world, program, x, swapped))
            predictions = (plant(program, x, entry) > 0).float()
            match = (predictions == y)
            exact += int(match.all(dim=-1).sum())
            bits += float(match.float().mean()) * 64
            total += 64
    return round(exact / total, 4), round(bits / total, 4)


def score_world(world: dict) -> dict:
    entry = entry_of(world)
    own_train = accuracy(world, train_programs, entry)
    own_held = accuracy(world, held_programs, entry)
    withheld = accuracy(world, held_programs, None)
    stranger = accuracy(world, held_programs, stranger_entry())
    swap = accuracy(world, held_programs, entry, swapped=True)
    return {"trained_programs": own_train[0],
            "held_programs": own_held[0],
            "withheld": withheld[0],
            "stranger": stranger[0],
            "swap_truth_with_own_entry": swap[0],
            "bits_trained_programs": own_train[1],
            "bits_held_programs": own_held[1],
            "bits_withheld": withheld[1],
            "bits_stranger": stranger[1]}


report = {
    "seed": args.seed, "width": W, "worlds": args.worlds,
    "held_worlds": args.held_worlds,
    "train_programs": len(train_programs),
    "held_programs": len(held_programs),
    "chance_exact": round(2.0 ** -W, 6), "chance_bit": 0.5,
    "held_out_worlds": {w["name"]: score_world(w) for w in held_worlds},
    "trained_worlds": {w["name"]: score_world(w)
                       for w in train_worlds[:args.held_worlds]},
}
print(json.dumps(report, indent=2))
if args.json:
    with open(args.json, "w") as handle:
        json.dump(report, handle, indent=2)

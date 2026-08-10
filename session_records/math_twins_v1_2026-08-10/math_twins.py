"""Number sequences as a fast proving ground for the reading mechanism.

F113 isolated the games' remaining failure to one question: can the
reader mint entries that a simple map SIGN-SPLITS — telling +rule from
-rule — or are twin entries too similar for anything downstream to
flip? Answering that in the games battery costs ~an hour per run. This
probe asks the identical question in a space with no search at all.

A world is a hidden affine rule over Z_M:

    x_next = (a * x + b) mod M

Its TWIN negates the additive part: (a, -b). The reader sees K example
(x, x_next) pairs and mints a bank entry; the frozen-architecture plant
predicts x_next for query x's it has not seen, conditioned only on
that entry. Everything else is the games recipe verbatim: ignorance
objective (entry-free prediction pushed to uniform), held-out worlds
split by twin PAIR, and the four controls — own entry, twin entry,
withheld entry, stranger entry.

Measurements, in the order they gate:
  * model level first: held-out accuracy per arm, twin entry cosine;
  * the F113 question directly: a linear probe trained on TRAIN-world
    entries to recover sign(b), scored on held-out entries. If this
    fails here, the reader is the defect and diff-entries the fix; if
    it succeeds here but the games' polarity scalar stays positive,
    the defect is the games value pathway instead.

Small on purpose: runs in minutes, so the fix loop is fast.
"""

from __future__ import annotations

import argparse
import json
import math

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=69316)
parser.add_argument("--modulus", type=int, default=16)
parser.add_argument("--dim", type=int, default=96)
parser.add_argument("--bank-tokens", type=int, default=8)
parser.add_argument("--examples", type=int, default=8,
                    help="K example (x, next) pairs shown to the reader")
parser.add_argument("--train-updates", type=int, default=8000)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--ignorance", type=float, default=0.5)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--pairs", type=int, default=24,
                    help="number of (b, -b) twin pairs; held-out pairs "
                         "are unseen in BOTH polarities")
parser.add_argument("--held-pairs", type=int, default=6)
parser.add_argument("--multiplicative", action="store_true",
                    help="draw a from units of Z_M too; default a=1 "
                         "keeps the rule purely additive so the twin "
                         "is a pure sign flip")
parser.add_argument("--json", default="")
args = parser.parse_args()

torch.manual_seed(args.seed)
M = args.modulus


def worlds() -> list[dict]:
    """Twin pairs (a, b) and (a, -b), b != 0 so twins differ."""
    generator = torch.Generator().manual_seed(args.seed * 7919)
    units = [u for u in range(1, M) if math.gcd(u, M) == 1]
    out = []
    for index in range(args.pairs):
        b = int(torch.randint(1, M, (1,), generator=generator))
        a = (units[int(torch.randint(0, len(units), (1,),
                                     generator=generator))]
             if args.multiplicative else 1)
        stem = f"a{a}b{b}"
        out.append({"name": stem, "a": a, "b": b, "stem": stem})
        out.append({"name": stem + "~", "a": a, "b": (M - b) % M,
                    "stem": stem})
    return out


def sample_pairs(world: dict, count: int,
                 generator: torch.Generator) -> tuple:
    x = torch.randint(0, M, (count,), generator=generator)
    return x, (world["a"] * x + world["b"]) % M


class Reader(torch.nn.Module):
    """K example (x, next) pairs -> bank entry, one forward pass."""

    def __init__(self, dim: int, tokens: int):
        super().__init__()
        self.embed = torch.nn.Embedding(M, dim)
        self.pair = torch.nn.Linear(2 * dim, dim)
        self.queries = torch.nn.Parameter(torch.randn(tokens, dim) * 0.02)
        self.blocks = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(
                dim, 4, dim_feedforward=2 * dim, batch_first=True,
                dropout=0.0, norm_first=True) for _ in range(2)])
        self.norm = torch.nn.LayerNorm(dim)

    def forward(self, x, nxt) -> torch.Tensor:
        rows = self.pair(torch.cat(
            [self.embed(x), self.embed(nxt)], dim=-1))
        token = torch.cat(
            [self.queries, rows], dim=0).unsqueeze(0)
        for block in self.blocks:
            token = block(token)
        return self.norm(token[0, :self.queries.shape[0]])


class Plant(torch.nn.Module):
    """(x, entry) -> next-value distribution. Frozen structure."""

    def __init__(self, dim: int):
        super().__init__()
        self.embed = torch.nn.Embedding(M, dim)
        self.blocks = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(
                dim, 4, dim_feedforward=2 * dim, batch_first=True,
                dropout=0.0, norm_first=True) for _ in range(2)])
        self.norm = torch.nn.LayerNorm(dim)
        self.head = torch.nn.Linear(dim, M)

    def forward(self, x, entry) -> torch.Tensor:
        token = self.embed(x).unsqueeze(1)
        if entry is not None:
            context = entry.unsqueeze(0).expand(x.shape[0], -1, -1)
            token = torch.cat([context, token], dim=1)
        for block in self.blocks:
            token = block(token)
        return self.head(self.norm(token[:, -1]))


reader = Reader(args.dim, args.bank_tokens)
plant = Plant(args.dim)
optimizer = torch.optim.Adam(
    list(reader.parameters()) + list(plant.parameters()), lr=args.lr)

all_worlds = worlds()
stems = sorted({w["stem"] for w in all_worlds})
select = torch.Generator().manual_seed(args.seed * 104729)
held_stems = set(s for s in [
    stems[int(i)] for i in torch.randperm(
        len(stems), generator=select)[:args.held_pairs]])
train = [w for w in all_worlds if w["stem"] not in held_stems]
held = [w for w in all_worlds if w["stem"] in held_stems]

data_gen = torch.Generator().manual_seed(args.seed * 15485863)
uniform = math.log(M)
for update in range(args.train_updates):
    world = train[int(torch.randint(0, len(train), (1,),
                                    generator=data_gen))]
    ex_x, ex_n = sample_pairs(world, args.examples, data_gen)
    q_x, q_n = sample_pairs(world, args.batch_size, data_gen)
    entry = reader(ex_x, ex_n)
    loss = torch.nn.functional.cross_entropy(plant(q_x, entry), q_n)
    if args.ignorance > 0:
        blind = plant(q_x, torch.zeros_like(entry)).log_softmax(-1)
        entropy = -(blind.exp() * blind).sum(-1).mean()
        loss = loss + args.ignorance * (uniform - entropy)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def entry_of(world: dict, offset: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(
        args.seed * 31 + hash(world["name"]) % 100000 + offset)
    ex_x, ex_n = sample_pairs(world, args.examples, generator)
    with torch.no_grad():
        return reader(ex_x, ex_n)


def twin_of(world: dict) -> dict:
    twin_name = (world["name"][:-1] if world["name"].endswith("~")
                 else world["name"] + "~")
    return next(w for w in all_worlds if w["name"] == twin_name)


def accuracy(world: dict, entry) -> float:
    generator = torch.Generator().manual_seed(args.seed * 977)
    q_x, q_n = sample_pairs(world, 256, generator)
    with torch.no_grad():
        logits = plant(q_x, entry)
    return round(float((logits.argmax(-1) == q_n).float().mean()), 4)


stranger_gen = torch.Generator().manual_seed(args.seed * 6700417)


def stranger_entry() -> torch.Tensor:
    b = int(torch.randint(1, M, (1,), generator=stranger_gen))
    return entry_of({"name": f"stranger{b}", "a": 1, "b": b}, offset=9)


def sign_probe() -> dict:
    """Linear probe: recover sign(b) from the entry — F113's question.

    Trained on train-world entries, scored on held-out entries. b in
    1..M-1; call b < M/2 'positive'. Uses concatenated tokens (F98:
    lossy pooling breaks addressing, so give the probe everything).
    """
    def flat(world):
        return entry_of(world).flatten()

    def label(world):
        return 1.0 if world["b"] < M / 2 else 0.0

    X = torch.stack([flat(w) for w in train])
    y = torch.tensor([label(w) for w in train])
    weight = torch.zeros(X.shape[1], requires_grad=True)
    bias = torch.zeros(1, requires_grad=True)
    probe_opt = torch.optim.Adam([weight, bias], lr=1e-2)
    for _ in range(500):
        logits = X @ weight + bias
        probe_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y)
        probe_opt.zero_grad()
        probe_loss.backward()
        probe_opt.step()
    with torch.no_grad():
        Xh = torch.stack([flat(w) for w in held])
        yh = torch.tensor([label(w) for w in held])
        predictions = (Xh @ weight + bias) > 0
        held_acc = float((predictions == yh.bool()).float().mean())
        train_acc = float(((X @ weight + bias > 0)
                           == y.bool()).float().mean())
    return {"train": round(train_acc, 4), "held": round(held_acc, 4)}


def twin_cosine(world: dict) -> float:
    a = entry_of(world).flatten()
    b = entry_of(twin_of(world)).flatten()
    return round(float(torch.nn.functional.cosine_similarity(
        a, b, dim=0)), 4)


report = {
    "seed": args.seed, "modulus": M, "pairs": args.pairs,
    "held_pairs": args.held_pairs, "examples": args.examples,
    "multiplicative": args.multiplicative,
    "held_out": {w["name"]: accuracy(w, entry_of(w)) for w in held},
    "twin_entry": {w["name"]: accuracy(w, entry_of(twin_of(w)))
                   for w in held},
    "withheld_entry": {w["name"]: accuracy(w, None) for w in held},
    "stranger_entry": {w["name"]: accuracy(w, stranger_entry())
                       for w in held},
    "twin_entry_cosine": {w["name"]: twin_cosine(w)
                          for w in held if not w["name"].endswith("~")},
    "sign_probe": sign_probe(),
}
print(json.dumps(report, indent=2))
if args.json:
    with open(args.json, "w") as handle:
        json.dump(report, handle, indent=2)

"""Which training domains are the compound lifts?

F206 found by accident what this probe looks for on purpose. Mixing
random parallel programs into the wake pool at 0.30 made the reader read
NEW worlds significantly for the first time, and nothing about grid
games or rule families predicted that a domain of pure noise would be
the thing that helped. If one domain can do that, the question is which
ones do it MOST, and whether that is a property you can measure in
advance rather than stumble on.

The analogy is the squat: a small number of movements that transfer to
everything, against a long tail of isolation work that transfers only to
itself. If task space has that structure, the training curriculum should
be chosen from the compound end and the rest is close to wasted compute.

METHOD.

  1. 113 domains, all speaking the same amodal slot interface and
     otherwise structurally unrelated -- pure copying, wrapping
     arithmetic, saturating arithmetic, conditionals, boolean worlds,
     shift registers, broadcast, sparse and dense programs, narrow value
     ranges, and the two REAL domains (rule families, grid games).
  2. Train one reader per domain, on that domain alone.
  3. Evaluate every reader on held-out worlds of every domain.
  4. T[i][j] = the fraction of the achievable margin that reader i
     recovers on domain j, where achievable means what the per-slot
     search gets and the floor means copying the input. Normalising this
     way is what makes domains with different difficulty comparable at
     all -- a raw fit of 0.9 means something different in a domain whose
     floor is 0.85 than in one whose floor is 0.15.
  5. Factor-analyse T. A general factor means transfer is mostly
     one-dimensional: some domains simply teach more. Several factors
     mean transfer is structured by KIND, and the curriculum needs one
     domain per factor rather than the top few overall.

  6. Then TEST the ranking rather than admiring it. Train fresh readers
     on the top-3 by generality, the bottom-3, three chosen at random,
     and all of them. If the ranking is real, top-3 beats bottom-3 and
     approaches all-domains at a fraction of the domains.

Scored with ground-truth execution throughout: the question is which
training distribution teaches reading, and routing it through the plant
would add the plant's error to every cell of the matrix.
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
parser.add_argument("--reader-updates", type=int, default=6000)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight-decay", type=float, default=0.01)
parser.add_argument("--examples", type=int, default=32)
parser.add_argument("--eval-rows", type=int, default=96)
parser.add_argument("--pool", type=int, default=1500)
parser.add_argument("--eval-worlds", type=int, default=48)
parser.add_argument("--combo-pool", type=int, default=1500)
parser.add_argument("--json", default="")
args = parser.parse_args()

torch.set_num_threads(1)
torch.manual_seed(args.seed)

SLOTS, VALUES = 6, 8
ABSENT = VALUES
HEIGHT = WIDTH = 8
PLANES = 3
PAR_OPS = ("NOOP", "INC", "DEC", "CINC", "CDEC", "COPY", "SINC", "SDEC")
MODULI = tuple(range(2, VALUES + 1))
NOOP = (0, 0, 0)
OP = {name: index for index, name in enumerate(PAR_OPS)}


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


def per_slot_search(before, after):
    program, cost = [], 0
    for s in range(SLOTS):
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


# =====================================================================
# the battery lives in `battery.py`: 113 domains spanning arithmetic,
# bitwise, order statistics, neighbourhood rules, indirection, state
# machines, aggregation, the plant's own program families, and the two
# real domains. Kept separate so this probe measures transfer and that
# module owns what the domains ARE.
# =====================================================================
from experiments.games_amodal.probes.battery import (        # noqa: E402
    REGISTRY as DOMAINS, names as domain_names)

NAMES = domain_names()


def stable(name: str) -> int:
    """A reproducible stand-in for the builtin string hash, which Python
    salts per process. Seeds derived from the builtin change between
    runs, which would make every number in this probe unreproducible."""
    total = 0
    for character in name:
        total = (total * 131 + ord(character)) % 1000003
    return total


# =====================================================================
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


def build_pool(domain_names, count, seed):
    """Wake on the given domains, labelled by the per-slot search."""
    rng = torch.Generator().manual_seed(seed)
    befores, afters, labels = [], [], []
    for index in range(count):
        name = domain_names[index % len(domain_names)]
        before, after = DOMAINS[name](rng, args.examples)
        befores.append(before)
        afters.append(after)
        labels.append([list(t) for t in per_slot_search(before, after)[0]])
    return (torch.stack(befores), torch.stack(afters),
            torch.tensor(labels))


def train_reader(pool, seed):
    torch.manual_seed(seed)
    reader = Reader(args.dim)
    opt = torch.optim.AdamW(reader.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    rng = torch.Generator().manual_seed(seed * 7919)
    for _ in range(args.reader_updates):
        take = torch.randint(0, pool[0].shape[0], (args.batch_size,),
                             generator=rng)
        before, after, labels = pool[0][take], pool[1][take], pool[2][take]
        po, pj, pm = reader(before, after)
        loss = (torch.nn.functional.cross_entropy(
                    po.reshape(-1, len(PAR_OPS)), labels[:, :, 0].reshape(-1))
                + torch.nn.functional.cross_entropy(
                    pj.reshape(-1, SLOTS), labels[:, :, 1].reshape(-1))
                + torch.nn.functional.cross_entropy(
                    pm.reshape(-1, len(MODULI)), labels[:, :, 2].reshape(-1)))
        opt.zero_grad()
        loss.backward()
        opt.step()
    for parameter in reader.parameters():
        parameter.requires_grad_(False)
    return reader


# ------------------------------------------------------ the eval sets
# Built ONCE. The floor and the search ceiling do not depend on which
# reader is being scored, so computing them per reader would multiply
# the cost of the matrix by the number of rows for no information.
EVAL = {}
for name in NAMES:
    rng = torch.Generator().manual_seed(args.seed + 424242 + stable(name) % 1000)
    worlds = []
    for _ in range(args.eval_worlds):
        fb, fa = DOMAINS[name](rng, args.examples)
        hb, ha = DOMAINS[name](rng, args.examples)          # a second draw is a second
        # world for the synthetic domains, which is the harsher reading
        # of "held out"; for the real domains it is a different world too
        moving = (hb != ha).any(dim=0)
        if int(moving.sum()) == 0:
            continue
        found, _ = per_slot_search(fb, fa)
        worlds.append({
            "fb": fb, "fa": fa, "hb": hb, "ha": ha, "moving": moving,
            "floor": float((hb[:, moving] == ha[:, moving]).float().mean()),
            "ceiling": float((run_parallel(hb, found)[:, moving]
                              == ha[:, moving]).float().mean())})
    EVAL[name] = worlds


def evaluate(reader, name):
    """One batched forward for the whole domain. Scoring world by world
    would mean 113 x 113 x eval_worlds separate forwards."""
    worlds = EVAL[name]
    if not worlds:
        return {"fit": None, "floor": None, "ceiling": None,
                "transfer": None}
    fb = torch.stack([w["fb"] for w in worlds])
    fa = torch.stack([w["fa"] for w in worlds])
    with torch.no_grad():
        po, pj, pm = reader(fb, fa)
    fits = []
    for index, world in enumerate(worlds):
        program = [(int(po[index, s].argmax()), int(pj[index, s].argmax()),
                    int(pm[index, s].argmax())) for s in range(SLOTS)]
        moving = world["moving"]
        fits.append(float((run_parallel(world["hb"], program)[:, moving]
                           == world["ha"][:, moving]).float().mean()))
    n = len(fits)
    fit = sum(fits) / n
    floor = sum(w["floor"] for w in worlds) / n
    ceiling = sum(w["ceiling"] for w in worlds) / n
    span = ceiling - floor
    return {"fit": round(fit, 4), "floor": round(floor, 4),
            "ceiling": round(ceiling, 4),
            "transfer": (round((fit - floor) / span, 4)
                         if span > 0.02 else None)}


report = {"seed": args.seed, "domains": NAMES,
          "eval_worlds": {n: len(EVAL[n]) for n in NAMES}}

matrix, cells = {}, {}
for source in NAMES:
    pool = build_pool([source], args.pool, args.seed * 31 + stable(source) % 997)
    reader = train_reader(pool, args.seed + stable(source) % 997)
    cells[source] = {target: evaluate(reader, target) for target in NAMES}
    matrix[source] = {t: cells[source][t]["transfer"] for t in NAMES}
report["cells"] = cells
report["transfer_matrix"] = matrix

# ------------------------------------------------- generality ranking
generality = {}
for source in NAMES:
    values = [matrix[source][t] for t in NAMES
              if t != source and matrix[source][t] is not None]
    generality[source] = round(sum(values) / len(values), 4) if values else None
report["generality_excluding_self"] = generality

# ------------------------------------------------------ factor analysis
usable = [t for t in NAMES
          if all(matrix[s][t] is not None for s in NAMES)]
M = torch.tensor([[matrix[s][t] for t in usable] for s in NAMES])
centred = M - M.mean(dim=0, keepdim=True)
scale = centred.std(dim=0, keepdim=True).clamp(min=1e-6)
Z = centred / scale
corr = (Z.T @ Z) / max(Z.shape[0] - 1, 1)
values, vectors = torch.linalg.eigh(corr)
order = torch.argsort(values, descending=True)
values, vectors = values[order], vectors[:, order]
total = float(values.clamp(min=0).sum())
report["factor_analysis"] = {
    "targets": usable,
    "eigenvalues": [round(float(v), 4) for v in values],
    "variance_explained": [round(float(v) / total, 4) for v in values],
    "factor1_loadings": {usable[i]: round(float(vectors[i, 0]), 4)
                         for i in range(len(usable))},
    "factor2_loadings": {usable[i]: round(float(vectors[i, 1]), 4)
                         for i in range(len(usable))},
    # each SOURCE's score on the general factor: how much it moves the
    # thing that all targets share
    "source_factor1_score": {
        NAMES[r]: round(float(Z[r] @ vectors[:, 0]), 4)
        for r in range(len(NAMES))}}

# =====================================================================
# the ROI test: is the ranking worth anything, or only descriptive?
# =====================================================================
ranked = sorted([n for n in NAMES if generality[n] is not None],
                key=lambda n: generality[n], reverse=True)
rng = torch.Generator().manual_seed(args.seed + 5)
random_three = [ranked[int(torch.randint(0, len(ranked), (1,),
                                         generator=rng))] for _ in range(3)]
arms = {
    "top3": ranked[:3],
    "bottom3": ranked[-3:],
    "random3": random_three,
    "all_domains": NAMES,
    # the curriculum the project has actually been using, as a baseline
    # for whether the ranking buys anything over "train on the real
    # domains you care about"
    "real_domains_only": [n for n in NAMES
                          if n.startswith("grid_") or n == "rule_families"],
}
roi = {}
for label, group in arms.items():
    pool = build_pool(group, args.combo_pool, args.seed * 37 + len(label))
    reader = train_reader(pool, args.seed + 77 + len(label))
    scores = {t: evaluate(reader, t)["transfer"] for t in NAMES}
    values_ = [v for v in scores.values() if v is not None]
    roi[label] = {"domains": group,
                  "mean_transfer_all_targets": round(
                      sum(values_) / len(values_), 4),
                  "per_target": scores}
report["roi_test"] = roi

print(json.dumps({k: v for k, v in report.items()
                  if k not in ("cells",)}, indent=2))
if args.json:
    with open(args.json, "w") as handle:
        json.dump(report, handle, indent=2)

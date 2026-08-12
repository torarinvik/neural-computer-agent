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
parser.add_argument("--split-offset", type=int, default=3,
                    help="which residue class mod 4 is held out. The "
                         "headline should not depend on WHICH quarter of "
                         "the battery is the test set; changing this is "
                         "the attack on that.")
parser.add_argument("--ranking-from", default="",
                    help="use ANOTHER run's generality ranking to choose "
                         "the top-k arms. The within-run ranking is fitted "
                         "on the same readers the arms are then judged "
                         "beside; a ranking imported from a different seed "
                         "has never seen this run at all, so it is the "
                         "out-of-sample test of whether the ranking "
                         "carries anything.")
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

# ---------------------------------------------------------------- SPLIT
# Every fourth domain is HELD OUT: no curriculum may train on it, and it
# is the only thing the ROI arms are scored on.
#
# Without this the comparison is rigged. An `all_domains` arm trains on
# every evaluation domain, so it is scored IN distribution while a
# 3-domain arm is scored out of distribution on 110 of 113 targets -- and
# the gap that produces is not transfer, it is the absence of a test set.
# The generality ranking is likewise computed on ELIGIBLE targets only,
# because a ranking fitted on the targets it will be judged against has
# seen the answer.
HELD_OUT = [n for i, n in enumerate(NAMES)
            if i % 4 == args.split_offset % 4]
ELIGIBLE = [n for n in NAMES if n not in HELD_OUT]


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
    values = [matrix[source][t] for t in ELIGIBLE
              if t != source and matrix[source][t] is not None]
    generality[source] = round(sum(values) / len(values), 4) if values else None
report["generality_excluding_self"] = generality
report["split"] = {"held_out": HELD_OUT, "eligible": ELIGIBLE}

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
ranking_source = generality
if args.ranking_from:
    with open(args.ranking_from) as handle:
        imported = json.load(handle)["generality_excluding_self"]
    ranking_source = {n: imported.get(n) for n in NAMES}
report["ranking_from"] = args.ranking_from or "self"
ranked = sorted([n for n in ELIGIBLE if ranking_source.get(n) is not None],
                key=lambda n: ranking_source[n], reverse=True)
rng = torch.Generator().manual_seed(args.seed + 5)


def sample_group(k):
    order = torch.randperm(len(ELIGIBLE), generator=rng)[:k]
    return [ELIGIBLE[int(i)] for i in order]


# Choosing for DIVERSITY rather than for generality. Each source's row of
# the transfer matrix is its profile -- what it teaches, target by
# target. Cluster the profiles and take one domain per cluster, so the
# group spans the space of things-that-can-be-taught instead of stacking
# several domains that teach the same thing well.
def spread_group(k, anchor=None):
    """Maximise distance between transfer PROFILES.

    `anchor` exists because spread starting from the best domain could
    be winning on the anchor alone -- top-1 plus nine pieces of filler.
    Starting from a domain chosen at random removes that explanation."""
    rows = {NAMES[r]: M[r] for r in range(len(NAMES))
            if NAMES[r] in ELIGIBLE}
    chosen = [anchor or ranked[0]]
    while len(chosen) < k:
        best, best_distance = None, -1.0
        for name in ELIGIBLE:
            if name in chosen or name not in rows:
                continue
            nearest = min(float((rows[name] - rows[c]).pow(2).sum())
                          for c in chosen)
            if nearest > best_distance:
                best, best_distance = name, nearest
        if best is None:
            break
        chosen.append(best)
    return chosen


arms = {
    "top3": ranked[:3],
    "bottom3": ranked[-3:],
    "random3": sample_group(3),
    "top10": ranked[:10],
    "random10": sample_group(10),
    "spread10": spread_group(10),
    "top25": ranked[:25],
    "random25": sample_group(25),
    "spread25": spread_group(25),
    "spread10_random_anchor": spread_group(
        10, anchor=ELIGIBLE[int(torch.randint(0, len(ELIGIBLE), (1,),
                                              generator=rng))]),
    "random50": sample_group(50),
    "all_eligible": ELIGIBLE,
    # the curriculum the project has actually been using, as a baseline
    # for whether any of this buys anything over "train on the real
    # domains you care about"
    "real_domains_only": [n for n in ELIGIBLE
                          if n.startswith("grid_") or n == "rule_families"],
}
roi = {}
for label, group in arms.items():
    pool = build_pool(group, args.combo_pool, args.seed * 37 + len(label))
    reader = train_reader(pool, args.seed + 77 + len(label))
    scores = {t: evaluate(reader, t)["transfer"] for t in NAMES}
    held = [scores[t] for t in HELD_OUT if scores[t] is not None]
    seen = [scores[t] for t in ELIGIBLE if scores[t] is not None]
    roi[label] = {"domains": group, "size": len(group),
                  # the number that counts: domains no arm trained on
                  "held_out_transfer": round(sum(held) / len(held), 4),
                  "eligible_transfer": round(sum(seen) / len(seen), 4),
                  "per_target": scores}
report["roi_test"] = roi

# =====================================================================
# ATTACKS ON THE RANKING ITSELF
#
# The ROI test above shows top-k beating random-k. That is consistent
# with the ranking carrying real information and ALSO consistent with
# two duller stories, so both get an arm.
#
#   SHUFFLED RANKING -- take the same ranking, permute it, and pick the
#     "top" three from the permuted order. If that scores like the real
#     top-3, the ranking is decoration and any three domains would do.
#   REDUNDANT TOP -- pick three domains that are individually strong but
#     have nearly IDENTICAL transfer profiles. If generality were all
#     that mattered these should match top-3; if the factor structure is
#     real they should lose to it, because they cover one factor three
#     times.
#   ANTI-SPREAD -- the three domains whose profiles are closest together
#     regardless of strength, as the floor of the same idea.
# =====================================================================
attack_rng = torch.Generator().manual_seed(args.seed + 31337)
shuffled = [ranked[int(i)] for i in
            torch.randperm(len(ranked), generator=attack_rng)]

rows_by_name = {NAMES[r]: M[r] for r in range(len(NAMES))}


def closest_trio(candidates):
    best, best_distance = None, None
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            for k in range(j + 1, len(candidates)):
                a_, b_, c_ = (candidates[i], candidates[j], candidates[k])
                if not all(n in rows_by_name for n in (a_, b_, c_)):
                    continue
                distance = (float((rows_by_name[a_] - rows_by_name[b_])
                                  .pow(2).sum())
                            + float((rows_by_name[a_] - rows_by_name[c_])
                                    .pow(2).sum())
                            + float((rows_by_name[b_] - rows_by_name[c_])
                                    .pow(2).sum()))
                if best_distance is None or distance < best_distance:
                    best, best_distance = [a_, b_, c_], distance
    return best


attack_arms = {
    "shuffled_ranking_top3": shuffled[:3],
    "redundant_top3": closest_trio(ranked[:12]) or ranked[:3],
    "anti_spread3": closest_trio(ranked) or ranked[:3],
}
attacks = {}
for label, group in attack_arms.items():
    pool = build_pool(group, args.combo_pool, args.seed * 41 + len(label))
    reader = train_reader(pool, args.seed + 91 + len(label))
    scores = {t: evaluate(reader, t)["transfer"] for t in NAMES}
    held = [scores[t] for t in HELD_OUT if scores[t] is not None]
    attacks[label] = {"domains": group,
                      "held_out_transfer": round(sum(held) / len(held), 4)}
report["attacks_on_ranking"] = attacks

print(json.dumps({k: v for k, v in report.items()
                  if k not in ("cells",)}, indent=2))
if args.json:
    with open(args.json, "w") as handle:
        json.dump(report, handle, indent=2)

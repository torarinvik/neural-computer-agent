"""Make the reader amodal, which F209's attack 3 showed it is not.

Everything under the reader is address-agnostic. The per-slot search is
permutation-equivariant by construction — permute the six slots and it
returns the permuted program, to four decimals. The plant is trained on
random programs over random states and has no notion of which slot is
which. Only the READER breaks: permuting slots costs it 0.9418 -> 0.5662
on rule families and 0.7920 -> 0.6053 on grid games.

The cause is in its shape. It pools the transitions into one vector and
reads off `SLOTS x 3` heads from it, so head number 4 is the thing that
predicts slot 4, and it learns that slot 0 is usually an avatar row.
That is a CONVENTION, and conventions do not survive relabelling.

The fix is structural rather than more training:

  1. **One token per slot, not one vector per world.** Each slot becomes
     a token carrying its own before/after value distribution.
  2. **Attention across slots**, which is permutation-EQUIVARIANT: permute
     the inputs and the outputs permute with them.
  3. **Shared heads.** One `op` head and one `modulus` head applied to
     every slot token, instead of six separate heads. A head that is
     indexed by slot position cannot be equivariant.
  4. **The `j` argument as a RELATION, not an index.** Predicting "slot
     4" is meaningless under permutation. Predicting "the slot my query
     attends to" is not, so `j` is a bilinear score between the slot
     token and every candidate slot token, which permutes correctly.

No slot embedding anywhere. The reader cannot know which slot it is
looking at, only what that slot does and how it relates to the others.

Both readers are trained on the same pool, from the same labels, and
scored on the same worlds — held out, and then held out AND permuted.
The permutation gap is the measurement.
"""

from __future__ import annotations

import argparse
import json

import torch

from experiments.games_amodal.probes.battery import (
    REGISTRY as DOMAINS, names as domain_names, run_parallel, slot_write,
    PAR_OPS, MODULI, SLOTS, VALUES)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=69316)
parser.add_argument("--dim", type=int, default=128)
parser.add_argument("--reader-updates", type=int, default=8000)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight-decay", type=float, default=0.01)
parser.add_argument("--examples", type=int, default=32)
parser.add_argument("--pool", type=int, default=3000)
parser.add_argument("--eval-worlds", type=int, default=24)
parser.add_argument("--heads", type=int, default=4)
parser.add_argument("--json", default="")
args = parser.parse_args()

torch.set_num_threads(1)
torch.manual_seed(args.seed)

NOOP = (0, 0, 0)
NAMES = domain_names()
HELD_OUT = [n for i, n in enumerate(NAMES) if i % 4 == 3]
ELIGIBLE = [n for n in NAMES if n not in HELD_OUT]


def per_slot_search(before, after):
    program = []
    for s in range(SLOTS):
        want = after[:, s]
        best, best_score = NOOP, -1.0
        for op in range(len(PAR_OPS)):
            for j in range(SLOTS):
                if j == s and PAR_OPS[op] in ("CINC", "CDEC", "COPY"):
                    continue
                for m in range(len(MODULI)):
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
    return program


# ------------------------------------------------------ the two readers
class IndexedReader(torch.nn.Module):
    """F205's reader, unchanged: pool everything, read off SLOTS x 3
    position-indexed heads."""

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


class EquivariantReader(torch.nn.Module):
    """One token per slot, attention across slots, shared heads, and the
    j argument as a relation.

    There is no slot embedding and no position-indexed parameter
    anywhere, so permuting the input slots permutes the output exactly.
    """

    def __init__(self, dim: int, heads: int):
        super().__init__()
        # a slot token is that slot's own value distribution before and
        # after, pooled over examples -- no identity, only behaviour
        self.token = torch.nn.Sequential(
            torch.nn.Linear(2 * VALUES + 2 * VALUES * VALUES, dim),
            torch.nn.ReLU(), torch.nn.Linear(dim, dim), torch.nn.ReLU())
        self.attend = torch.nn.MultiheadAttention(dim, heads,
                                                  batch_first=True)
        self.norm1 = torch.nn.LayerNorm(dim)
        self.mix = torch.nn.Sequential(
            torch.nn.Linear(dim, 2 * dim), torch.nn.ReLU(),
            torch.nn.Linear(2 * dim, dim))
        self.norm2 = torch.nn.LayerNorm(dim)
        # SHARED across slots: applied to every token with the same
        # weights, which is what makes them equivariant
        self.op = torch.nn.Linear(dim, len(PAR_OPS))
        self.arg_m = torch.nn.Linear(dim, len(MODULI))
        # j is scored as a relation between the writing slot and each
        # candidate slot, so it permutes with the slots instead of
        # naming one
        self.query = torch.nn.Linear(dim, dim)
        self.key = torch.nn.Linear(dim, dim)

    def tokens(self, before, after):
        b, e, _ = before.shape
        hot_b = torch.nn.functional.one_hot(before, VALUES).float()
        hot_a = torch.nn.functional.one_hot(after, VALUES).float()
        # marginals, and the JOINT before-value x after-value table per
        # slot, which is what says "this slot increments" without
        # reference to which slot it is
        joint = torch.einsum("besi,besj->besij", hot_b, hot_a).mean(dim=1)
        pair = torch.einsum("besi,besj->besij", hot_a, hot_b).mean(dim=1)
        feature = torch.cat([
            hot_b.mean(dim=1), hot_a.mean(dim=1),
            joint.reshape(b, SLOTS, -1), pair.reshape(b, SLOTS, -1)], dim=-1)
        return self.token(feature)

    def forward(self, before, after):
        latent = self.tokens(before, after)
        attended, _ = self.attend(latent, latent, latent, need_weights=False)
        latent = self.norm1(latent + attended)
        latent = self.norm2(latent + self.mix(latent))
        scores = torch.einsum("bsd,btd->bst", self.query(latent),
                              self.key(latent)) / (latent.shape[-1] ** 0.5)
        return self.op(latent), scores, self.arg_m(latent)


def build_pool(count, seed):
    rng = torch.Generator().manual_seed(seed)
    befores, afters, labels = [], [], []
    for index in range(count):
        name = ELIGIBLE[index % len(ELIGIBLE)]
        before, after = DOMAINS[name](rng, args.examples)
        befores.append(before)
        afters.append(after)
        labels.append([list(t) for t in per_slot_search(before, after)])
    return (torch.stack(befores), torch.stack(afters), torch.tensor(labels))


def train(reader, pool, seed):
    opt = torch.optim.AdamW(reader.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    rng = torch.Generator().manual_seed(seed)
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


def read(reader, before, after):
    with torch.no_grad():
        po, pj, pm = reader(before.unsqueeze(0), after.unsqueeze(0))
    return [(int(po[0, s].argmax()), int(pj[0, s].argmax()),
             int(pm[0, s].argmax())) for s in range(SLOTS)]


pool = build_pool(args.pool, args.seed * 15485863)
readers = {
    "indexed": train(IndexedReader(args.dim), pool, args.seed * 104729),
    "equivariant": train(EquivariantReader(args.dim, args.heads), pool,
                         args.seed * 104729),
}
report = {"seed": args.seed, "pool": int(pool[0].shape[0]),
          "held_out_domains": len(HELD_OUT),
          "parameters": {k: sum(p.numel() for p in r.parameters())
                         for k, r in readers.items()}}

# ------------------------------------------------------- the evaluation
rows: dict = {k: {"plain": [], "permuted": [], "floor": [], "search": [],
                  "search_permuted": []} for k in readers}
eval_rng = torch.Generator().manual_seed(args.seed + 31337)
perm_rng = torch.Generator().manual_seed(args.seed + 909)
for name in HELD_OUT:
    for _ in range(args.eval_worlds):
        fb, fa = DOMAINS[name](eval_rng, args.examples)
        hb, ha = DOMAINS[name](eval_rng, args.examples)
        moving = (hb != ha).any(dim=0)
        if int(moving.sum()) == 0:
            continue
        order = torch.randperm(SLOTS, generator=perm_rng)
        pfb, pfa, phb, pha = (fb[:, order], fa[:, order],
                              hb[:, order], ha[:, order])
        pmoving = moving[order]

        def score(program, b, a, mask):
            return float((run_parallel(b, program)[:, mask]
                          == a[:, mask]).float().mean())

        searched = per_slot_search(fb, fa)
        searched_perm = per_slot_search(pfb, pfa)
        for key, reader in readers.items():
            rows[key]["plain"].append(score(read(reader, fb, fa), hb, ha,
                                            moving))
            rows[key]["permuted"].append(score(read(reader, pfb, pfa), phb,
                                               pha, pmoving))
            rows[key]["floor"].append(
                float((hb[:, moving] == ha[:, moving]).float().mean()))
            rows[key]["search"].append(score(searched, hb, ha, moving))
            rows[key]["search_permuted"].append(
                score(searched_perm, phb, pha, pmoving))

summary = {}
for key, data in rows.items():
    def mean(field):
        return round(sum(data[field]) / len(data[field]), 4)
    summary[key] = {
        "n": len(data["plain"]), "identity_floor": mean("floor"),
        "held_out": mean("plain"), "held_out_permuted": mean("permuted"),
        "permutation_gap": round(mean("plain") - mean("permuted"), 4),
        "search": mean("search"), "search_permuted": mean("search_permuted")}
report["summary"] = summary

# F209 measured the permutation failure on the REAL domains, where slot 0
# is always an avatar row and a convention therefore exists to be learned.
# The battery's synthetic domains carry no such convention, so the gap
# nearly vanishes there for both readers and would understate the fix.
# This scores the domains the attack actually used.
real_names = [n for n in NAMES if n.startswith("grid_") or n == "rule_families"]
real_rng = torch.Generator().manual_seed(args.seed + 5150)
real: dict = {k: {"plain": [], "permuted": [], "floor": []} for k in readers}
for name in real_names:
    for _ in range(args.eval_worlds):
        fb, fa = DOMAINS[name](real_rng, args.examples)
        hb, ha = DOMAINS[name](real_rng, args.examples)
        moving = (hb != ha).any(dim=0)
        if int(moving.sum()) == 0:
            continue
        order = torch.randperm(SLOTS, generator=real_rng)
        for key, reader in readers.items():
            got = read(reader, fb, fa)
            real[key]["plain"].append(
                float((run_parallel(hb, got)[:, moving]
                       == ha[:, moving]).float().mean()))
            got_p = read(reader, fb[:, order], fa[:, order])
            real[key]["permuted"].append(
                float((run_parallel(hb[:, order], got_p)[:, moving[order]]
                       == ha[:, order][:, moving[order]]).float().mean()))
            real[key]["floor"].append(
                float((hb[:, moving] == ha[:, moving]).float().mean()))
report["real_domains"] = {
    key: {"n": len(v["plain"]),
          "identity_floor": round(sum(v["floor"]) / len(v["floor"]), 4),
          "plain": round(sum(v["plain"]) / len(v["plain"]), 4),
          "permuted": round(sum(v["permuted"]) / len(v["permuted"]), 4),
          "permutation_gap": round(
              (sum(v["plain"]) - sum(v["permuted"])) / len(v["plain"]), 4)}
    for key, v in real.items()}

# Equivariance is a property of the FUNCTION, not of a score, so check it
# directly: feed the same world twice, once permuted, and see whether the
# emitted programs are the same program up to that permutation.
check_rng = torch.Generator().manual_seed(args.seed + 4242)
exact = {k: 0 for k in readers}
total = 0
for name in HELD_OUT[:12]:
    for _ in range(8):
        fb, fa = DOMAINS[name](check_rng, args.examples)
        order = torch.randperm(SLOTS, generator=check_rng)
        inverse = torch.argsort(order)
        total += 1
        for key, reader in readers.items():
            plain = read(reader, fb, fa)
            permuted = read(reader, fb[:, order], fa[:, order])
            # the program read from the permuted world, mapped back
            mapped = [(permuted[int(inverse[s])][0],
                       int(order[permuted[int(inverse[s])][1]]),
                       permuted[int(inverse[s])][2]) for s in range(SLOTS)]
            exact[key] += int(mapped == plain)
report["equivariance_check"] = {
    k: {"identical_after_relabelling": exact[k], "of": total,
        "rate": round(exact[k] / max(total, 1), 4)} for k in readers}

print(json.dumps(report, indent=2))
if args.json:
    with open(args.json, "w") as handle:
        json.dump(report, handle, indent=2)

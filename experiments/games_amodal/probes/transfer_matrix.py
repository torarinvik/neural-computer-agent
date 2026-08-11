"""Pairwise transfer, measured for every source-target pair.

The founding objective is "having learned task A makes a NOVEL task B
faster to learn than from scratch". Every probe so far has measured
that in AGGREGATE — pre-train on a POOL of families, test on held-out
ones (F75-F82) — which answers "does diversity help" but never "which
task donates". A transfer MATRIX answers the objective directly, one
ordered pair at a time, and it is the measurement the objective
literally names.

The design constraint that makes it affordable: **train once per
source, evaluate against every target.** N sources, N targets, but
only N+2 training runs, because evaluation freezes the plant and fits
only a small per-family bank entry. That is the F75 protocol, and it
is also the architecture's own claim — if a plant trained on A helps
on B, it must show up with A's weights FROZEN and only B's entry
learned, or the help was just warm-started weights and not transfer
through structure.

    T[i][j] = held-out accuracy on family j, using a plant trained
              only on family i, frozen, with a fresh entry for j.

Two rows are controls rather than sources:

  * `none`      — an untrained plant. This is the from-scratch
                  baseline every T[i][j] must beat to mean anything.
  * `scrambled` — trained on a schema-DESTROYED family of the same
                  size and action count. If T[scrambled][j] is as high
                  as T[i][j], then "any training helps" and the
                  structure of the source is irrelevant. Without this
                  row the whole matrix could be a warm-start artefact.

BREADTH over depth, deliberately: the families are chosen to share as
little as possible — hand-made line/dial/toggle/perm, procedurally
generated ones (plain, wide, gated), spatial grid and walled, and
chaos which has no rule at all. They share only the amodal SLOTS x
VALUES interface, which is the point: whatever transfers across THAT
gap is structure rather than subject matter.

What the matrix supports once built:
  * row mean   — how much training on i helps everything else. High =
                 a high-ROI curriculum task, which is the quantity the
                 project actually wants and has never measured.
  * col mean   — how much j benefits from anything. High = an easy
                 target, not evidence about any source.
  * SVD        — whether transfer is one general factor or several
                 distinct skills. One dominant component would say
                 there is a single "general competence" axis; several
                 would say curricula should be assembled, not ranked.
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
parser.add_argument("--bank-tokens", type=int, default=8)
parser.add_argument("--source-updates", type=int, default=6000,
                    help="budget for training the plant on ONE source")
parser.add_argument("--entry-updates", type=int, default=400,
                    help="budget for fitting a target's entry against "
                         "the FROZEN plant. Deliberately small: this "
                         "measures what the plant makes easy, not what "
                         "the entry can learn on its own.")
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--procedural", type=int, default=6,
                    help="procedurally generated families to include")
parser.add_argument("--json", default="")
args = parser.parse_args()

torch.manual_seed(args.seed)


def build_families() -> list:
    """As structurally unlike each other as this interface allows."""
    out = [("line", Family("line")), ("dial", Family("dial")),
           ("toggle", Family("toggle")), ("perm", Family("perm")),
           ("grid", Family("grid")), ("walled", Family("walled")),
           ("chaos", Family("chaos"))]
    generator = torch.Generator().manual_seed(args.seed * 7919)
    for index in range(args.procedural):
        flavour = index % 3
        spec = random_family_spec(
            generator, wide=(flavour == 1), gated=(flavour == 2))
        tag = ("plain", "wide", "gated")[flavour]
        out.append((f"proc{index}_{tag}", RandomFamily(spec)))
    return out


families = build_families()
names = [n for n, _ in families]


def rollout(family, count: int, generator: torch.Generator):
    size = len(family.states)
    state = torch.randint(0, size, (count,), generator=generator)
    action = torch.randint(0, family.actions, (count,),
                           generator=generator)
    nxt = torch.tensor([family.table[int(s)][int(a)]
                        for s, a in zip(state, action)])
    return (family.slot_values(state), action,
            family.slot_values(nxt))


def train_plant(family, updates: int, seed: int) -> SlotModel:
    """Train a plant on ONE family. No bank entry: the source's content
    goes into the weights, which is what makes the transfer question
    meaningful — whatever survives to a new family is structure."""
    plant = SlotModel(args.dim)
    if updates == 0:
        return plant
    optimizer = torch.optim.Adam(plant.parameters(), lr=args.lr)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(updates):
        values, action, target = rollout(family, args.batch_size,
                                         generator)
        logits = plant(values, action)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, VALUES), target.reshape(-1),
            ignore_index=VALUES)      # VALUES marks an unused slot
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return plant


def fit_entry(plant: SlotModel, family, seed: int) -> float:
    """FREEZE the plant, fit only this family's bank entry, and report
    held-out next-state accuracy. Entry-only adaptation is the
    architecture's own claim, so transfer that does not appear here is
    not transfer the architecture can use."""
    for parameter in plant.parameters():
        parameter.requires_grad_(False)
    entry = torch.zeros(args.bank_tokens, args.dim, requires_grad=True)
    torch.nn.init.normal_(entry, std=0.02)
    optimizer = torch.optim.Adam([entry], lr=1e-2)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(args.entry_updates):
        values, action, target = rollout(family, args.batch_size,
                                         generator)
        logits = plant(values, action, bank=entry)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, VALUES), target.reshape(-1),
            ignore_index=VALUES)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    held = torch.Generator().manual_seed(seed + 99991)
    values, action, target = rollout(family, 512, held)
    with torch.no_grad():
        got = plant(values, action, bank=entry).argmax(-1)
    used = target != VALUES
    return round(float((got[used] == target[used]).float().mean()), 4)


rows: dict = {}

# control row: no training at all — the from-scratch baseline
rows["none"] = {}
base_plant = train_plant(families[0][1], 0, args.seed)
for name, family in families:
    rows["none"][name] = fit_entry(base_plant, family, args.seed * 31)

# control row: trained on a schema-destroyed family. If this donates as
# much as a real source, the matrix is measuring warm start, not
# structure.
scrambled = Family("line", scramble=True, seed=args.seed)
rows["scrambled"] = {}
plant = train_plant(scrambled, args.source_updates, args.seed * 13)
for name, family in families:
    rows["scrambled"][name] = fit_entry(plant, family, args.seed * 31)

for source_name, source in families:
    plant = train_plant(source, args.source_updates,
                        args.seed * 17 + abs(hash(source_name)) % 9973)
    rows[source_name] = {}
    for target_name, target in families:
        rows[source_name][target_name] = fit_entry(
            plant, target, args.seed * 31)

# advantage over the from-scratch row, which is the founding objective
# stated as a number: does having learned A help on B?
advantage = {
    source: {target: round(value - rows["none"][target], 4)
             for target, value in targets.items()}
    for source, targets in rows.items()}


def mean(values) -> float:
    values = list(values)
    return round(sum(values) / max(len(values), 1), 4)


donor = {source: mean(v for t, v in targets.items() if t != source)
         for source, targets in advantage.items()
         if source not in ("none",)}
receptivity = {
    target: mean(advantage[s][target] for s in rows
                 if s not in ("none", "scrambled") and s != target)
    for target in names}

report = {
    "seed": args.seed, "families": names,
    "source_updates": args.source_updates,
    "entry_updates": args.entry_updates,
    "accuracy": rows,
    "advantage_over_scratch": advantage,
    "donor_strength": donor,
    "target_receptivity": receptivity,
}
print(json.dumps(report, indent=2))
if args.json:
    with open(args.json, "w") as handle:
        json.dump(report, handle, indent=2)

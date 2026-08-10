"""Frozen structural plant + external content bank.

The experiment F73/F74 forced, with its prediction recorded in advance
in `docs/MEMORY_BANK_DESIGN.md`.

F73: a slot-symmetric plant learns these families 2.36x cheaper than a
dense one, and 1.03x when the structure is scrambled away -- so the gain
is structure, causally. F74: the same plant is WORSE than cold when the
families arrive in sequence (380 vs 280) and retention sits at the
chance floor, because the weights holding structure are the same weights
being asked to hold content.

The split those two findings imply is the project's founding
architecture: structure in the plant, content in the bank. Here it is,
measured rather than asserted.

  * pre-train the slot plant on THREE families, each family's content
    carried by its own small bank entry (K context tokens the plant
    reads through attention)
  * FREEZE every plant weight
  * learn the fourth, held-out family by fitting a fresh bank entry
    ALONE -- no weight ever moves again

Leave-one-out over all four families, so no single lucky pairing can
carry the result.

Predicted (recorded before running): retention flat, because two bank
entries are separate tensors and cannot overwrite one another; cost for
the held-out family below its cold cost, because structure is already
paid for; no negative transfer, because nothing shared can conflict.

Three nulls, because a bank result is exactly the kind that flatters
itself:

  random-plant   the plant is frozen at INITIALISATION, never
                 pre-trained. Only the bank entry is fitted. If this
                 matches the pre-trained plant, pre-training bought
                 nothing and the result is really about the bank
                 entry's own capacity.
  scrambled      the plant is pre-trained on SCRAMBLED versions of the
                 three families -- same sizes, same interfaces, schema
                 destroyed -- then frozen. If this matches the real
                 plant, the transfer is not structure. This is the same
                 control that decided F72 and F73.
  cold-full      a fresh slot model with every weight trainable and no
                 bank at all: F73's baseline, the number the bank arm
                 has to beat to mean anything.

Cost is updates ACTUALLY SPENT to reach `--stop-at` exhaustive dynamics
accuracy. Pre-training cost is reported separately and is an investment
amortised over every later family, never hidden.
"""

from __future__ import annotations

import argparse
import json

import torch

from experiments.games_amodal.probes.schema_families import (
    FAMILIES, Family, RandomFamily, SlotModel, random_family_spec)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=69316)
parser.add_argument("--held-out", default="perm", choices=tuple(FAMILIES))
parser.add_argument("--dim", type=int, default=64)
parser.add_argument("--bank-tokens", type=int, default=4,
                    help="K context tokens per family -- the bank entry")
parser.add_argument("--pretrain-updates", type=int, default=1500)
parser.add_argument(
    "--pretrain-families", type=int, default=0,
    help="pre-train on N families SAMPLED FROM THE SCHEMA instead of the "
         "three hand-made ones. The first bank run measured why this "
         "matters: with three fixed families the plant learns three "
         "modes rather than how to read an entry, and a fourth entry has "
         "nothing general to plug into (held-out accuracy 0.069 against "
         "a cold 1.000). A distribution makes reading the entry the only "
         "strategy that generalises. The four hand-made families are "
         "never sampled, so they stay held out.")
parser.add_argument("--updates", type=int, default=600,
                    help="budget cap for the held-out family")
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--stop-at", type=float, default=0.98)
parser.add_argument("--lr", type=float, default=3e-3)
parser.add_argument("--bank-lr", type=float, default=1e-2,
                    help="bank entries are a handful of tokens, not a "
                         "network, so they take a larger step")
parser.add_argument("--json", default="")
args = parser.parse_args()

torch.manual_seed(args.seed)

HELD_OUT = args.held_out
PRETRAIN = [name for name in FAMILIES if name != HELD_OUT]


def new_entry() -> torch.Tensor:
    return torch.nn.Parameter(torch.randn(args.bank_tokens, args.dim) * 0.02)


def accuracy(plant, family: Family, entry) -> float:
    """Exhaustive over every (state, action) pair of this family."""
    size = len(family.states)
    states = torch.arange(size).repeat_interleave(family.actions)
    acts = torch.arange(family.actions).repeat(size)
    with torch.no_grad():
        logits = plant(family.slot_values(states), acts, entry)
    blocks = logits.argmax(-1)
    nxt = torch.tensor([family.table[int(s)][int(a)]
                        for s, a in zip(states, acts)])
    want = family.slot_targets(nxt)
    correct = torch.ones(states.shape[0], dtype=torch.bool)
    for slot in range(family.slots):
        correct &= blocks[:, slot] == want[:, slot]
    return round(float(correct.float().mean()), 4)


def batch_loss(plant, family: Family, entry, generator) -> torch.Tensor:
    size = len(family.states)
    states = torch.randint(0, size, (args.batch_size,), generator=generator)
    acts = torch.randint(0, family.actions, (args.batch_size,),
                         generator=generator)
    nxt = torch.tensor([family.table[int(s)][int(a)]
                        for s, a in zip(states, acts)])
    logits = plant(family.slot_values(states), acts, entry)
    want = family.slot_targets(nxt)
    total = 0.0
    for slot in range(family.slots):
        total = total + torch.nn.functional.cross_entropy(
            logits[:, slot, :], want[:, slot])
    return total / family.slots


def pretrain(families: dict, names=None) -> tuple:
    """Plant weights + one bank entry per pre-training family, jointly.

    Content is pushed into the entries by construction: the plant sees
    every family and can only tell them apart through the entry it is
    given, so whatever the weights keep has to be common to all three.
    """
    names = list(names if names is not None else families)
    plant = SlotModel(args.dim)
    entries = {name: new_entry() for name in names}
    optimizer = torch.optim.Adam(
        [{"params": plant.parameters(), "lr": args.lr},
         {"params": list(entries.values()), "lr": args.bank_lr}])
    generator = torch.Generator().manual_seed(args.seed + 31)
    spent = args.pretrain_updates
    check = names[:8]
    for step in range(args.pretrain_updates):
        name = names[step % len(names)]
        loss = batch_loss(plant, families[name], entries[name], generator)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step + 1) % 50 == 0:
            scores = [accuracy(plant, families[n], entries[n])
                      for n in check]
            if sum(scores) / len(scores) >= args.stop_at:
                spent = step + 1
                break
    for parameter in plant.parameters():
        parameter.requires_grad_(False)
    return plant, entries, spent


def learn_entry(plant, family: Family) -> tuple:
    """Fit a bank entry with EVERY plant weight frozen."""
    entry = new_entry()
    optimizer = torch.optim.Adam([entry], lr=args.bank_lr)
    generator = torch.Generator().manual_seed(args.seed + 77)
    best = accuracy(plant, family, entry)
    if best >= args.stop_at:
        return entry, best, 0
    for step in range(args.updates):
        loss = batch_loss(plant, family, entry, generator)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step + 1) % 25 == 0:
            best = accuracy(plant, family, entry)
            if best >= args.stop_at:
                return entry, best, step + 1
    return entry, accuracy(plant, family, entry), args.updates


def learn_full(family: Family) -> tuple:
    """F73's cold baseline: fresh plant, all weights trainable, no bank."""
    plant = SlotModel(args.dim)
    optimizer = torch.optim.Adam(plant.parameters(), lr=args.lr)
    generator = torch.Generator().manual_seed(args.seed + 77)
    for step in range(args.updates):
        loss = batch_loss(plant, family, None, generator)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step + 1) % 25 == 0:
            score = accuracy(plant, family, None)
            if score >= args.stop_at:
                return score, step + 1
    return accuracy(plant, family, None), args.updates


real = {name: Family(name, False, args.seed) for name in FAMILIES}
scrambled = {name: Family(name, True, args.seed) for name in FAMILIES}

report = {"seed": args.seed, "held_out": HELD_OUT, "pretrain": PRETRAIN,
          "bank_tokens": args.bank_tokens, "stop_at": args.stop_at,
          "budget": args.updates}

# --- the arm under test: structure pre-trained and FROZEN, content banked
if args.pretrain_families:
    generator = torch.Generator().manual_seed(args.seed + 5)
    sampled = {f"proc{i}": RandomFamily(random_family_spec(generator))
               for i in range(args.pretrain_families)}
    source, source_names = sampled, list(sampled)
    scrambled_source = {
        name: RandomFamily(family.spec, name) for name, family in
        sampled.items()}
    for position, (name, family) in enumerate(scrambled_source.items()):
        size = len(family.states)
        shuffle = torch.Generator().manual_seed(args.seed + 11 + position)
        columns = [torch.randperm(size, generator=shuffle)
                   for _ in range(family.actions)]
        family.table = [[int(column[row]) for column in columns]
                        for row in range(size)]
else:
    source, source_names = real, PRETRAIN
    scrambled_source = scrambled
report["pretrain_family_count"] = len(source_names)
plant, entries, pretrain_cost = pretrain(source, source_names)
report["pretrain_cost"] = pretrain_cost
report["pretrain_accuracy"] = {
    name: accuracy(plant, source[name], entries[name])
    for name in source_names[:8]}
# snapshot BEFORE the held-out entry is fitted, so retention is a delta
report["retention_before"] = dict(report["pretrain_accuracy"])
entry, held_accuracy, held_cost = learn_entry(plant, real[HELD_OUT])
report["bank_pretrained"] = {"cost": held_cost, "accuracy": held_accuracy}

# Retention: the three pre-training families, re-checked AFTER the
# held-out entry was fitted. Separate tensors cannot overwrite one
# another, so this is the architectural guarantee -- measured, because
# an interface bug could still leak content into the plant.
report["retention_after"] = {name: accuracy(plant, source[name],
                                            entries[name])
                             for name in source_names[:8]}
report["retention_delta"] = {
    name: round(report["retention_after"][name]
                - report["retention_before"][name], 4)
    for name in report["retention_before"]}

# --- null 1: was pre-training necessary, or is the bank entry doing it?
random_plant = SlotModel(args.dim)
for parameter in random_plant.parameters():
    parameter.requires_grad_(False)
_, random_accuracy, random_cost = learn_entry(random_plant, real[HELD_OUT])
report["bank_random_plant"] = {"cost": random_cost,
                               "accuracy": random_accuracy}

# --- null 2: was it STRUCTURE, or just any pre-training?
scrambled_plant, _, scrambled_pretrain = pretrain(
    scrambled_source, list(scrambled_source) if args.pretrain_families
    else PRETRAIN)
_, scrambled_accuracy, scrambled_cost = learn_entry(
    scrambled_plant, real[HELD_OUT])
report["bank_scrambled_plant"] = {"cost": scrambled_cost,
                                  "accuracy": scrambled_accuracy,
                                  "pretrain_cost": scrambled_pretrain}

# --- the baseline it has to beat
cold_accuracy, cold_cost = learn_full(real[HELD_OUT])
report["cold_full"] = {"cost": cold_cost, "accuracy": cold_accuracy}

report["trainable_parameters"] = {
    "cold_full": sum(p.numel() for p in SlotModel(args.dim).parameters()),
    "bank_entry": args.bank_tokens * args.dim}

print(json.dumps(report, indent=2))
if args.json:
    with open(args.json, "w") as handle:
        json.dump(report, handle, indent=2)

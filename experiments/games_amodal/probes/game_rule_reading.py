"""Amortised rule reading, on the actual games battery.

F71-F98 established the mechanism on procedurally generated transition
families and on the reacher expressed in a slot interface. Everything
the GAMES add — real screens, verifier-private rules, reward instead of
next-state — has been untested since F70. This probe closes that.

The `dual` variant is the games' own factorisation test and it maps onto
the reading mechanism exactly. Each trial puts the avatar at the centre
with `arity` items on adjacent cells and a cue drawn across the top row.
For cue k, exactly one side is edible (+1); the others cost -1. The rule
pair (rule0, rule1) is verifier-private: the screen shows WHICH marks
are present and which cue is active, never which mark is food.

  * a "family" here is one (rule0, rule1) pairing — with arity 3 there
    are 9 pairings built from 6 independent sub-rules;
  * the reader watches a handful of (screen, action, reward) triples and
    emits a bank entry;
  * the plant, frozen at test, predicts the reward of each action from
    the screen and that entry; behaviour is argmax over actions.

This is the same mechanism as F76-F98 with one substitution: the model
predicts REWARD rather than NEXT STATE. That substitution matters and is
the point. F67 concluded "store a model, not a policy" because a policy
is preferential and goes stale. A game's difficulty is preferential —
which item to eat — so the question is whether a rule can be read as a
FACT about outcomes rather than stored as a habit. "Side 1 is edible
under cue 0" is factual, checkable, and cannot go stale; "move right" is
a habit that the next variant contradicts.

Held-out pairings are the test. A pairing whose two sub-rules each
appear in training under DIFFERENT partners is novel as a whole and
familiar in its parts, which is precisely the compounding claim.

Nulls, in the form the earlier findings taught:
  withheld   entry zeroed — the plant alone must not know the rule
  stranger   another pairing's entry, drawn at random rather than from a
             neighbour (F93: positional pairing stops being a control)
  random     plant frozen at initialisation, never trained
"""

from __future__ import annotations

import argparse
import itertools
import json

import torch

import dataclasses

from experiments.games_amodal.game_family import (
    FamilyConfig, FamilyVerifier, _DELTAS, family_variants)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=69316)
parser.add_argument("--arity", type=int, default=3)
parser.add_argument("--dim", type=int, default=64)
parser.add_argument("--bank-tokens", type=int, default=8)
parser.add_argument("--context", type=int, default=64,
                    help="observed (screen, action, reward) triples")
parser.add_argument("--train-updates", type=int, default=6000)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--trials", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--holdout", type=int, default=3,
                    help="pairings held out entirely")
parser.add_argument(
    "--variants", action="store_true",
    help="use the battery's OWN variant enumeration as the family "
         "distribution instead of dual rule pairings. F99 measured the "
         "dual game giving only 6 training pairings, and the verifier "
         "caps `arity` at 3, so 9 pairings is a HARD ceiling on that "
         "axis. F78 says memorisation is what a distribution that small "
         "should produce. `family_variants` x `inverted` gives 50 "
         "distinct worlds, which is the largest rule diversity the "
         "battery can supply without changing the games themselves.")
parser.add_argument("--random-plant", action="store_true")
parser.add_argument("--json", default="")
args = parser.parse_args()

torch.manual_seed(args.seed)
ACTIONS = 4
PLANES, HEIGHT, WIDTH = 3, 8, 8
SCREEN = PLANES * HEIGHT * WIDTH


def make_verifier(rule0: int, rule1: int, seed: int) -> FamilyVerifier:
    config = FamilyConfig(dual=1, arity=args.arity, rule0=rule0, rule1=rule1,
                          name=f"dual{rule0}{rule1}")
    verifier = FamilyVerifier(config, batch_size=args.batch_size, seed=seed)
    verifier.reset(seed=seed)
    return verifier


def truth_actions(verifier: FamilyVerifier) -> torch.Tensor:
    """Which action reaches the edible item, per row. Harness-side only.

    The learner never sees this; it exists so mastery can be scored
    against the rule the verifier is actually enforcing rather than
    against reward alone.
    """
    out = torch.zeros(verifier.batch_size, dtype=torch.long)
    for row in range(verifier.batch_size):
        edible = verifier.config.edible(verifier._dual_kind[row])
        centre = verifier._avatar[row]
        for item in verifier._dual_items[row]:
            if item[2] != edible:
                continue
            delta = (item[0] - centre[0], item[1] - centre[1])
            for index, candidate in enumerate(_DELTAS):
                if candidate == delta:
                    out[row] = index
    return out


def observe(verifier: FamilyVerifier, count: int, generator) -> tuple:
    """Roll trials, acting at random, and record what happened."""
    screens, acts, rewards = [], [], []
    while len(screens) * verifier.batch_size < count:
        screen = verifier.observation().reshape(verifier.batch_size, -1)
        action = torch.randint(0, ACTIONS, (verifier.batch_size,),
                               generator=generator)
        step = verifier.step(action)
        screens.append(screen)
        acts.append(action)
        rewards.append(step.reward)
    screens = torch.cat(screens)[:count]
    acts = torch.cat(acts)[:count]
    rewards = torch.cat(rewards)[:count]
    # three outcome classes: cost, nothing, food
    labels = torch.ones_like(rewards, dtype=torch.long)
    labels[rewards > 0.05] = 2
    labels[rewards < -0.05] = 0
    return screens, acts, labels


class Reader(torch.nn.Module):
    """Observed outcomes -> a bank entry, in one forward pass."""

    def __init__(self, dim: int, tokens: int):
        super().__init__()
        self.tokens = tokens
        self.embed = torch.nn.Linear(SCREEN + ACTIONS + 3, dim)
        self.queries = torch.nn.Parameter(torch.randn(tokens, dim) * 0.02)
        self.blocks = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(
                dim, 4, dim_feedforward=2 * dim, batch_first=True,
                dropout=0.0, norm_first=True) for _ in range(2)])
        self.norm = torch.nn.LayerNorm(dim)

    def forward(self, screens, acts, labels) -> torch.Tensor:
        rows = torch.cat([
            screens,
            torch.nn.functional.one_hot(acts, ACTIONS).float(),
            torch.nn.functional.one_hot(labels, 3).float()], dim=-1)
        x = torch.cat([self.queries, self.embed(rows)], dim=0).unsqueeze(0)
        for block in self.blocks:
            x = block(x)
        return self.norm(x[0, :self.tokens])


class Plant(torch.nn.Module):
    """(screen, action, entry) -> outcome class. Factual, not preferential.

    It answers "what happens if", never "what should I do". Behaviour is
    derived by taking the action whose predicted outcome is best, which
    is the same policy-free derivation F67 established for dynamics.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.screen = torch.nn.Linear(SCREEN, dim)
        self.action = torch.nn.Embedding(ACTIONS, dim)
        self.blocks = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(
                dim, 4, dim_feedforward=2 * dim, batch_first=True,
                dropout=0.0, norm_first=True) for _ in range(2)])
        self.norm = torch.nn.LayerNorm(dim)
        self.head = torch.nn.Linear(dim, 3)

    def forward(self, screens, acts, entry) -> torch.Tensor:
        token = (self.screen(screens) + self.action(acts)).unsqueeze(1)
        if entry is not None:
            context = entry.unsqueeze(0).expand(screens.shape[0], -1, -1)
            token = torch.cat([context, token], dim=1)
        for block in self.blocks:
            token = block(token)
        return self.head(self.norm(token[:, -1]))


def variant_configs() -> list:
    """Every valid world the battery can enumerate, inverted and not."""
    out = []
    for base in family_variants(max_components=2, levels=(1, 2)):
        for inverted in (False, True):
            config = dataclasses.replace(
                base, inverted=inverted,
                name=f"{base.name}{'~' if inverted else ''}")
            try:
                config.validate()
            except ValueError:
                continue
            out.append(config)
    return out


PAIRINGS = list(itertools.product(range(args.arity), repeat=2))
shuffle = torch.randperm(len(PAIRINGS),
                         generator=torch.Generator().manual_seed(args.seed))
held_out = [PAIRINGS[int(i)] for i in shuffle[:args.holdout]]
train_pairs = [p for p in PAIRINGS if p not in held_out]

def verifier_for(config: FamilyConfig, seed: int) -> FamilyVerifier:
    verifier = FamilyVerifier(config, batch_size=args.batch_size, seed=seed)
    verifier.reset(seed=seed)
    return verifier


def play(config: FamilyConfig, entry, seed: int,
         random_actions: bool = False) -> float:
    """Mean reward per step. For most variants there is no single
    'correct' action, so reward against a measured random floor is the
    honest score -- the same accounting the battery itself uses."""
    verifier = verifier_for(config, seed)
    generator = torch.Generator().manual_seed(seed + 5)
    total = 0.0
    for _ in range(args.trials):
        screen = verifier.observation().reshape(verifier.batch_size, -1)
        if random_actions:
            choice = torch.randint(0, ACTIONS, (verifier.batch_size,),
                                   generator=generator)
        else:
            scores = []
            for action in range(ACTIONS):
                column = torch.full((verifier.batch_size,), action,
                                    dtype=torch.long)
                logits = plant(screen, column, entry).softmax(-1)
                scores.append(logits[:, 2] - logits[:, 0])
            choice = torch.stack(scores, dim=1).argmax(dim=1)
        total += float(verifier.step(choice).reward.mean())
    return round(total / args.trials, 4)


plant, reader = Plant(args.dim), Reader(args.dim, args.bank_tokens)
optimizer = torch.optim.Adam(
    list(plant.parameters()) + list(reader.parameters()), lr=args.lr)
generator = torch.Generator().manual_seed(args.seed + 31)

if not args.random_plant:
    # one live verifier per training pairing, rolled continuously: making
    # a fresh one every update re-deals from scratch and costs far more
    # than it buys, and the trials are i.i.d. either way
    live = {pair: make_verifier(pair[0], pair[1], args.seed + index)
            for index, pair in enumerate(train_pairs)}
    for step in range(args.train_updates):
        rule0, rule1 = train_pairs[step % len(train_pairs)]
        verifier = live[(rule0, rule1)]
        screens, acts, labels = observe(verifier, args.context, generator)
        entry = reader(screens, acts, labels)
        query_s, query_a, query_l = observe(verifier, args.batch_size,
                                            generator)
        loss = torch.nn.functional.cross_entropy(
            plant(query_s, query_a, entry), query_l)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(plant.parameters()) + list(reader.parameters()), 1.0)
        optimizer.step()

for parameter in list(plant.parameters()) + list(reader.parameters()):
    parameter.requires_grad_(False)


def evaluate(rule0: int, rule1: int, *, entry_from=None,
             withheld: bool = False) -> dict:
    """Read the rule, then choose by predicted outcome. Zero training."""
    probe = torch.Generator().manual_seed(args.seed + 900)
    verifier = make_verifier(rule0, rule1, args.seed + 7000)
    screens, acts, labels = observe(verifier, args.context, probe)
    if withheld:
        entry = torch.zeros(args.bank_tokens, args.dim)
    elif entry_from is not None:
        other = make_verifier(entry_from[0], entry_from[1], args.seed + 11)
        entry = reader(*observe(other, args.context, probe))
    else:
        entry = reader(screens, acts, labels)

    correct, outcome = 0, 0.0
    fresh = make_verifier(rule0, rule1, args.seed + 8000)
    for _ in range(args.trials):
        screen = fresh.observation().reshape(fresh.batch_size, -1)
        truth = truth_actions(fresh)
        scores = []
        for action in range(ACTIONS):
            column = torch.full((fresh.batch_size,), action,
                                dtype=torch.long)
            logits = plant(screen, column, entry).softmax(-1)
            scores.append(logits[:, 2] - logits[:, 0])
        choice = torch.stack(scores, dim=1).argmax(dim=1)
        correct += int((choice == truth).sum())
        step = fresh.step(choice)
        outcome += float(step.reward.mean())
    total = args.trials * fresh.batch_size
    return {"choice_accuracy": round(correct / total, 4),
            "mean_reward": round(outcome / args.trials, 4)}


if args.variants:
    configs = variant_configs()
    order = torch.randperm(len(configs),
                           generator=torch.Generator().manual_seed(args.seed))
    cut = max(4, len(configs) // 4)
    held = [configs[int(i)] for i in order[:cut]]
    train = [configs[int(i)] for i in order[cut:]]
    optimizer = torch.optim.Adam(
        list(plant.parameters()) + list(reader.parameters()), lr=args.lr)
    for parameter in list(plant.parameters()) + list(reader.parameters()):
        parameter.requires_grad_(True)
    generator = torch.Generator().manual_seed(args.seed + 31)
    if not args.random_plant:
        live = {index: verifier_for(config, args.seed + index)
                for index, config in enumerate(train)}
        for step in range(args.train_updates):
            index = step % len(train)
            verifier = live[index]
            entry = reader(*observe(verifier, args.context, generator))
            query_s, query_a, query_l = observe(verifier, args.batch_size,
                                                generator)
            loss = torch.nn.functional.cross_entropy(
                plant(query_s, query_a, entry), query_l)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(plant.parameters()) + list(reader.parameters()), 1.0)
            optimizer.step()
    for parameter in list(plant.parameters()) + list(reader.parameters()):
        parameter.requires_grad_(False)

    probe = torch.Generator().manual_seed(args.seed + 900)

    def score(config: FamilyConfig, mode: str) -> dict:
        if mode == "withheld":
            entry = torch.zeros(args.bank_tokens, args.dim)
        elif mode == "stranger":
            other = train[int(torch.randint(0, len(train), (1,),
                                            generator=probe))]
            entry = reader(*observe(verifier_for(other, args.seed + 21),
                                    args.context, probe))
        else:
            entry = reader(*observe(verifier_for(config, args.seed + 31),
                                    args.context, probe))
        return {"reward": play(config, entry, args.seed + 8000),
                "floor": play(config, entry, args.seed + 8000,
                              random_actions=True)}

    report = {"seed": args.seed, "mode": "variants",
              "train_count": len(train), "held_out_count": len(held),
              "random_plant": args.random_plant,
              "held_out": {c.name: score(c, "read") for c in held},
              "trained": {c.name: score(c, "read") for c in train[:len(held)]},
              "withheld_entry": {c.name: score(c, "withheld") for c in held},
              "stranger_entry": {c.name: score(c, "stranger") for c in held}}
    print(json.dumps(report, indent=2))
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(report, handle, indent=2)
    raise SystemExit(0)

report = {"seed": args.seed, "arity": args.arity,
          "train_pairings": [list(p) for p in train_pairs],
          "held_out_pairings": [list(p) for p in held_out],
          "random_plant": args.random_plant,
          "chance": round(1.0 / ACTIONS, 4)}
report["held_out"] = {f"{a}{b}": evaluate(a, b) for a, b in held_out}
report["trained"] = {f"{a}{b}": evaluate(a, b)
                     for a, b in train_pairs[:len(held_out)]}
report["withheld_entry"] = {f"{a}{b}": evaluate(a, b, withheld=True)
                            for a, b in held_out}
report["stranger_entry"] = {
    f"{a}{b}": evaluate(a, b, entry_from=next(
        p for p in train_pairs if p != (a, b) and p[0] != a and p[1] != b))
    for a, b in held_out}

print(json.dumps(report, indent=2))
if args.json:
    with open(args.json, "w") as handle:
        json.dump(report, handle, indent=2)

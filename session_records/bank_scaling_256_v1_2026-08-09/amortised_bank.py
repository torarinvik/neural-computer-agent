"""Amortised bank entries: acquire a family by READING it, not fitting it.

F75 left exactly one gap. The frozen-plant/banked-content split solved
forgetting outright (retention delta 0.0000 over 96 measurements) and
showed causal structure transfer (held-out accuracy 0.973 schema-
pretrained vs 0.626 scrambled vs 0.083 random plant), but acquiring a
novel family still cost about TWICE a cold fit in updates -- 123 against
62, cheaper in only 2 of 12 runs. That remaining cost is gradient
descent inferring content which a few dozen observed transitions already
determine.

This probe removes the gradient descent. An ENCODER maps a handful of
observed (state, action, next state) triples directly to a bank entry,
trained across the family distribution so that reading is the only
strategy that generalises. At test time on a novel family the cost is
one forward pass: ZERO gradient steps, zero weights moved.

  * a pool of families sampled from the schema (the four hand-made
    families are never sampled, so they stay genuinely held out)
  * each update: draw a family, encode M context transitions into an
    entry, and predict DISJOINT query transitions of that same family
  * the plant and the encoder train together; at test both are frozen

Predicted, recorded in `docs/MEMORY_BANK_DESIGN.md` before this ran:
cost per novel family drops below cold by an order of magnitude,
retention stays exactly flat, and the random-plant and scrambled nulls
stay dead. If amortised entries plateau below 0.98 on held-out families,
the entry is too weak a channel to carry content and the bank needs a
richer interface than context tokens.

Nulls. An in-context claim is the easiest kind to fake, so:

  wrong-context  the encoder is fed transitions from a DIFFERENT family
                 than the one being predicted. If accuracy holds up, the
                 encoder is not reading anything -- it has memorised the
                 distribution and the entry is decoration. This is the
                 null that matters most here.
  random-plant   plant and encoder frozen at initialisation.
  scrambled      trained on scrambled dynamics -- same sizes, same
                 interfaces, schema destroyed. The control that decided
                 F72, F73 and F75.
  cold-full      a fresh slot model, every weight trainable, no bank:
                 the 62-update baseline the amortised arm must beat.

Cost accounting is deliberately unkind to this probe: pre-training is
reported in full and never amortised away, and the amortised arm is
credited with ZERO only for genuinely fitting nothing at test time.
"""

from __future__ import annotations

import argparse
import json

import torch

from experiments.games_amodal.probes.schema_families import (
    ACTIONS, FAMILIES, SLOTS, VALUES, Family, RandomFamily,
    SlotModel, random_family_spec)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=69316)
parser.add_argument("--dim", type=int, default=64)
parser.add_argument("--bank-tokens", type=int, default=16)
parser.add_argument("--context", type=int, default=64,
                    help="M observed transitions the encoder reads")
parser.add_argument("--pool", type=int, default=256,
                    help="families sampled from the schema to train on. "
                         "F75 measured the failure mode this guards "
                         "against: with three fixed families the plant "
                         "learns three MODES instead of how to read an "
                         "entry (held-out 0.069 against a cold 1.000).")
parser.add_argument("--train-updates", type=int, default=6000)
parser.add_argument("--query", type=int, default=64)
parser.add_argument("--layers", type=int, default=2)
parser.add_argument("--heads", type=int, default=4)
parser.add_argument(
    "--film", action="store_true",
    help="entry also emits per-family gains/biases that modulate the "
         "shared blocks, instead of only being prepended as tokens. "
         "F76: the narrow channel, not the reading, is what caps "
         "acquisition. Weights stay shared and frozen; per-family "
         "parameters stay in the bank, so retention must stay 0.0000 -- "
         "and if it does not, any channel wide enough to be expressive "
         "is wide enough to interfere.")
parser.add_argument("--updates", type=int, default=600,
                    help="budget cap for the fine-tune comparison")
parser.add_argument("--stop-at", type=float, default=0.98)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--bank-lr", type=float, default=1e-2)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--scramble", action="store_true",
                    help="control: train on scrambled dynamics")
parser.add_argument(
    "--wide", action="store_true",
    help="widen the generator's SUPPORT with simultaneous two-slot ops "
         "and permutation state spaces. F79 measured a hard floor the "
         "schema itself caused: `toggle` read at 0.096 at every pool "
         "size because it flips a PAIR of bits and no op could express "
         "that, and `perm`'s states are permutations rather than a "
         "product. Diversity within a schema cannot buy coverage "
         "outside it -- only the schema can.")
parser.add_argument(
    "--sequential", type=int, default=0,
    help="acquire N novel families ONE AFTER ANOTHER through the frozen "
         "plant, keeping every entry, and report cost against position "
         "plus retention of all earlier entries. This turns the open "
         "framing question -- is lifetime cost over a bounded task "
         "count the right gate? -- into a measurement. Break-even at "
         "~936 families (F82) is an EXTRAPOLATION from 16, and it is "
         "only valid if the per-task saving is stable as the bank "
         "grows. If cost drifts upward with bank size, or retention "
         "decays, then break-even is a mirage and the gate is "
         "ill-formed. If both stay flat, the extrapolation holds and "
         "936 is simply a finite number of tasks.")
parser.add_argument(
    "--retrieval", action="store_true",
    help="after growing the bank, FIND the right entry among N instead "
         "of being handed it. F83: clause (c) of the primary gate "
         "cannot fail while entries are handed to the correct family by "
         "construction, because nothing then scales with bank size. "
         "Retrieval is the cost that does. Scores every stored entry by "
         "how well it predicts a few HELD-OUT transitions of the task "
         "at hand and takes the best -- F44's consequence probing, "
         "which is the project's own answer to worlds that look alike.")
parser.add_argument("--novel-count", type=int, default=16,
                    help="held-out in-support families to evaluate")
parser.add_argument("--probe-transitions", type=int, default=32,
                    help="held-out transitions used to identify a task")
parser.add_argument("--random-plant", action="store_true",
                    help="control: skip training entirely")
parser.add_argument("--json", default="")
args = parser.parse_args()

torch.manual_seed(args.seed)

TRIPLE = SLOTS * (VALUES + 1) * 2 + ACTIONS


class EntryEncoder(torch.nn.Module):
    """Observed transitions -> a bank entry, in one forward pass.

    K learned query tokens attend over the M transition tokens; their
    outputs ARE the entry. Nothing about a family is stored in these
    weights -- the encoder holds how to READ a family, which is
    structure, while what it emits is content.
    """

    def __init__(self, dim: int, tokens: int, heads: int = 4,
                 layers: int = 2):
        super().__init__()
        self.tokens = tokens
        self.embed = torch.nn.Linear(TRIPLE, dim)
        self.queries = torch.nn.Parameter(torch.randn(tokens, dim) * 0.02)
        self.blocks = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(
                dim, heads, dim_feedforward=2 * dim, batch_first=True,
                dropout=0.0, norm_first=True)
            for _ in range(layers)])
        self.norm = torch.nn.LayerNorm(dim)

    def forward(self, triples: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.queries, self.embed(triples)], dim=0)
        x = x.unsqueeze(0)
        for block in self.blocks:
            x = block(x)
        return self.norm(x[0, :self.tokens])


def triple_features(family: Family, states: torch.Tensor,
                    acts: torch.Tensor,
                    nxt: torch.Tensor) -> torch.Tensor:
    """One row per observed transition: state, action, next state."""
    def slots_one_hot(indices: torch.Tensor) -> torch.Tensor:
        values = family.slot_values(indices)
        return torch.nn.functional.one_hot(
            values, VALUES + 1).float().reshape(indices.shape[0], -1)
    action = torch.zeros(acts.shape[0], ACTIONS)
    action.scatter_(1, acts.unsqueeze(-1), 1.0)
    return torch.cat([slots_one_hot(states), action, slots_one_hot(nxt)],
                     dim=-1)


def sample(family: Family, count: int, generator) -> tuple:
    size = len(family.states)
    states = torch.randint(0, size, (count,), generator=generator)
    acts = torch.randint(0, family.actions, (count,), generator=generator)
    nxt = torch.tensor([family.table[int(s)][int(a)]
                        for s, a in zip(states, acts)])
    return states, acts, nxt


def context_entry(encoder, family: Family, generator) -> torch.Tensor:
    states, acts, nxt = sample(family, args.context, generator)
    return encoder(triple_features(family, states, acts, nxt))


def query_loss(plant, family: Family, entry, generator) -> torch.Tensor:
    states, acts, nxt = sample(family, args.query, generator)
    logits = plant(family.slot_values(states), acts, entry)
    want = family.slot_targets(nxt)
    total = 0.0
    for slot in range(family.slots):
        total = total + torch.nn.functional.cross_entropy(
            logits[:, slot, :], want[:, slot])
    return total / family.slots


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


def build_pool() -> list:
    generator = torch.Generator().manual_seed(args.seed + 5)
    pool = []
    for index in range(args.pool):
        family = RandomFamily(
            random_family_spec(generator, wide=args.wide), f"proc{index}")
        if args.scramble:
            size = len(family.states)
            shuffle = torch.Generator().manual_seed(args.seed + 11 + index)
            columns = [torch.randperm(size, generator=shuffle)
                       for _ in range(family.actions)]
            family.table = [[int(column[row]) for column in columns]
                            for row in range(size)]
        pool.append(family)
    return pool


def train(pool: list) -> tuple:
    plant = SlotModel(args.dim, args.heads, args.layers, args.film)
    encoder = EntryEncoder(args.dim, args.bank_tokens,
                           args.heads, args.layers)
    if args.random_plant:
        return plant, encoder, 0
    optimizer = torch.optim.Adam(
        list(plant.parameters()) + list(encoder.parameters()), lr=args.lr)
    generator = torch.Generator().manual_seed(args.seed + 31)
    for step in range(args.train_updates):
        family = pool[step % len(pool)]
        entry = context_entry(encoder, family, generator)
        loss = query_loss(plant, family, entry, generator)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(plant.parameters()) + list(encoder.parameters()), 1.0)
        optimizer.step()
    return plant, encoder, args.train_updates


def finetune(plant, family: Family, entry: torch.Tensor) -> tuple:
    """Cost to reach stop-at, STARTING from the amortised entry.

    Comparable with F75's 123 updates from a random entry and the cold
    62. Zero means the read alone already cleared the bar.
    """
    working = torch.nn.Parameter(entry.detach().clone())
    if accuracy(plant, family, working) >= args.stop_at:
        return 0, accuracy(plant, family, working), working.detach()
    optimizer = torch.optim.Adam([working], lr=args.bank_lr)
    generator = torch.Generator().manual_seed(args.seed + 77)
    for step in range(args.updates):
        loss = query_loss(plant, family, working, generator)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step + 1) % 25 == 0:
            score = accuracy(plant, family, working)
            if score >= args.stop_at:
                return step + 1, score, working.detach()
    return (args.updates, accuracy(plant, family, working),
            working.detach())


def cold_full(family: Family) -> tuple:
    plant = SlotModel(args.dim, args.heads, args.layers)
    optimizer = torch.optim.Adam(plant.parameters(), lr=3e-3)
    generator = torch.Generator().manual_seed(args.seed + 77)
    for step in range(args.updates):
        states, acts, nxt = sample(family, args.batch_size, generator)
        logits = plant(family.slot_values(states), acts, None)
        want = family.slot_targets(nxt)
        loss = sum(torch.nn.functional.cross_entropy(
            logits[:, slot, :], want[:, slot])
            for slot in range(family.slots)) / family.slots
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step + 1) % 25 == 0:
            score = accuracy(plant, family, None)
            if score >= args.stop_at:
                return step + 1, score
    return args.updates, accuracy(plant, family, None)


pool = build_pool()
plant, encoder, train_cost = train(pool)
for parameter in list(plant.parameters()) + list(encoder.parameters()):
    parameter.requires_grad_(False)

report = {"seed": args.seed, "film": args.film, "wide": args.wide, "pool": args.pool, "context": args.context,
          "bank_tokens": args.bank_tokens, "train_cost": train_cost,
          "scramble": args.scramble, "random_plant": args.random_plant,
          "stop_at": args.stop_at, "held_out": {}}

# In-distribution readout FIRST. If the encoder cannot read families it
# was trained on, a zero on the held-out families means "broken", not
# "does not generalise", and the two are indistinguishable without this.
probe_generator = torch.Generator().manual_seed(args.seed + 500)
report["in_distribution"] = round(sum(
    accuracy(plant, family, context_entry(encoder, family, probe_generator))
    for family in pool[:16]) / 16, 4)

# NOVEL families from the same generator, never in the training pool.
# This separates two questions the hand-made families confound. `toggle`
# flips a PAIR of bits and `perm`'s states are permutations rather than a
# product space -- neither op is in the generator's vocabulary, so those
# families are outside the training distribution's SUPPORT, not merely
# unseen instances of it. A family that is unseen but in-support is the
# honest test of "task A makes novel task B cheaper"; the hand-made four
# additionally test transfer beyond the schema that was trained on, which
# is a strictly harder claim and is reported separately.
novel_generator = torch.Generator().manual_seed(args.seed + 4242)
novel = [RandomFamily(random_family_spec(novel_generator, wide=args.wide),
                      f"novel{i}")
         for i in range(args.novel_count)]
seen = {tuple(sorted(f.spec["ops"])) + (f.spec["values"],) for f in pool}
novel = [f for f in novel
         if tuple(sorted(f.spec["ops"])) + (f.spec["values"],) not in seen]
novel_read, novel_tune, novel_cold = [], [], []
for family in novel:
    entry = context_entry(encoder, family, novel_generator)
    novel_read.append(accuracy(plant, family, entry))
    novel_tune.append(finetune(plant, family, entry)[0])
    novel_cold.append(cold_full(family)[0])
# Per-family diagnostics. `toggle` has been the worst-read family at
# every configuration (0.527 at its best, F84), and "toggle is hard" is
# not actionable. Recording each novel family's SHAPE alongside its read
# accuracy turns it into "families with property X are hard", which is.
report["novel_detail"] = [
    {"slots": f.spec["slots"], "values": f.spec["values"],
     "actions": f.spec["actions"], "space": f.spec.get("space", "product"),
     "ops": sorted({o[0] for o in f.spec["ops"]}),
     "states": len(f.states), "read": r}
    for f, r in zip(novel, novel_read)]
report["novel_in_support"] = {
    "count": len(novel),
    "read_accuracy": round(sum(novel_read) / len(novel_read), 4),
    "read_cost": 0,
    "mastered_by_reading": sum(1 for a in novel_read if a >= args.stop_at),
    "finetune_cost": round(sum(novel_tune) / len(novel_tune), 1),
    "cold_cost": round(sum(novel_cold) / len(novel_cold), 1)}

held = {name: Family(name, False, args.seed) for name in FAMILIES}
generator = torch.Generator().manual_seed(args.seed + 900)
entries = {}
for name, family in held.items():
    entry = context_entry(encoder, family, generator)
    entries[name] = entry
    read_accuracy = accuracy(plant, family, entry)
    tune_cost, tune_accuracy, _ = finetune(plant, family, entry)
    cold_cost, cold_accuracy = cold_full(family)
    report["held_out"][name] = {
        "read_accuracy": read_accuracy, "read_cost": 0,
        "finetune_cost": tune_cost, "finetune_accuracy": tune_accuracy,
        "cold_cost": cold_cost, "cold_accuracy": cold_accuracy}

# WITHHELD-BANK arm, the missing half of the double dissociation.
# The wrong-context null shows a CORRUPTED entry fails; this shows an
# ABSENT one does. Together they establish that the capability lives in
# the bank and not in the frozen weights -- present -> mastery,
# withheld -> chance, weights alone retain nothing. Imported from the
# protocol used in the parallel Codex session, which had this arm where
# this probe did not.
zero_entry = torch.zeros(args.bank_tokens, args.dim)
report["withheld_bank"] = {
    "novel_in_support": round(sum(
        accuracy(plant, family, zero_entry) for family in novel)
        / max(len(novel), 1), 4),
    "hand_made": {name: accuracy(plant, held[name], zero_entry)
                  for name in held}}

# Null: feed the encoder ANOTHER family's transitions. If accuracy
# survives, nothing is being read and the entry is decoration.
if args.sequential:
    # One frozen plant, N families in sequence, every entry kept.
    growth_generator = torch.Generator().manual_seed(args.seed + 8888)
    grown, history = {}, []
    cumulative_bank = 0
    cumulative_cold = 0
    for position in range(args.sequential):
        family = RandomFamily(
            random_family_spec(growth_generator, wide=args.wide),
            f"grow{position}")
        entry = context_entry(encoder, family, growth_generator)
        read = accuracy(plant, family, entry)
        cost, final, tuned = finetune(plant, family, entry)
        cold_cost, _ = cold_full(family)
        grown[position] = (family, tuned, entry.detach().clone())
        cumulative_bank += cost
        cumulative_cold += cold_cost
        history.append({"position": position, "read": read, "cost": cost,
                        "accuracy": final, "cold_cost": cold_cost,
                        "cumulative_bank": cumulative_bank,
                        "cumulative_cold": cumulative_cold})
    # Retention across the WHOLE grown bank, after every entry exists.
    drift = []
    for position, (family, entry, _) in grown.items():
        drift.append(round(accuracy(plant, family, entry)
                           - history[position]["accuracy"], 4))
    report["sequential"] = {
        "count": args.sequential, "history": history,
        "retention_drift_max": max(abs(d) for d in drift),
        "retention_drift_mean": round(sum(drift) / len(drift), 4)}

if args.sequential and args.retrieval:
    # ---- RETRIEVAL: identify the task by consequence, not by fiat ----
    def entry_score(family: Family, entry, states, acts, nxt) -> float:
        """How well does THIS entry explain these observed transitions?"""
        with torch.no_grad():
            logits = plant(family.slot_values(states), acts, entry)
        blocks = logits.argmax(-1)
        want = family.slot_targets(nxt)
        correct = torch.ones(states.shape[0], dtype=torch.bool)
        for slot in range(family.slots):
            correct &= blocks[:, slot] == want[:, slot]
        return float(correct.float().mean())

    retrieval_generator = torch.Generator().manual_seed(args.seed + 5150)
    retrieval = {}
    for size in (8, 16, 32, 64, 128, 256):
        if size > len(grown):
            continue
        pool_entries = [grown[i][1] for i in range(size)]
        pool_addresses = [grown[i][2] for i in range(size)]
        hits, margins, in_bank_best = 0, [], []
        for target in range(size):
            family = grown[target][0]
            states, acts, nxt = sample(family, args.probe_transitions,
                                       retrieval_generator)
            scores = [entry_score(family, e, states, acts, nxt)
                      for e in pool_entries]
            best = max(range(size), key=lambda j: scores[j])
            hits += int(best == target)
            in_bank_best.append(scores[best])
            ordered = sorted(scores, reverse=True)
            margins.append(ordered[0] - (ordered[1] if size > 1 else 0.0))
        # DISCRIMINATION NULL: a task that is NOT in the bank must score
        # LOW against every entry. Without this, "retrieval accuracy" is
        # satisfied by a system that always returns something.
        outside = []
        for _ in range(16):
            stranger = RandomFamily(
                random_family_spec(retrieval_generator, wide=args.wide))
            states, acts, nxt = sample(stranger, args.probe_transitions,
                                       retrieval_generator)
            outside.append(max(entry_score(stranger, e, states, acts, nxt)
                               for e in pool_entries))
        # ---- CONTENT-ADDRESSED retrieval, for comparison ----
        # F85 measured the failure this answers: a linear scan costs N
        # PLANT forward passes, so at N=64 recognising a task is dearer
        # than minting one. Here a fresh read is compressed to one key
        # and matched against stored keys by cosine -- ONE encoder pass
        # and an N x d matmul, with no plant forwards at all. Optionally
        # the top-k candidates are then verified by consequence, which
        # costs k plant passes and is CONSTANT in N.
        def key_of(entry: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.normalize(entry.mean(0), dim=-1)

        keys = torch.stack([key_of(a) for a in pool_addresses])
        key_hits, verify_hits, key_in, key_out = 0, 0, [], []
        topk = min(4, size)
        for target in range(size):
            family = grown[target][0]
            query = key_of(context_entry(encoder, family,
                                         retrieval_generator))
            similarity = keys @ query
            key_hits += int(int(similarity.argmax()) == target)
            key_in.append(float(similarity.max()))
            # retrieve-then-verify: k plant passes, constant in N
            candidates = torch.topk(similarity, topk).indices.tolist()
            states, acts, nxt = sample(family, args.probe_transitions,
                                       retrieval_generator)
            best = max(candidates,
                       key=lambda j: entry_score(family, pool_entries[j],
                                                 states, acts, nxt))
            verify_hits += int(best == target)
        for _ in range(16):
            stranger = RandomFamily(
                random_family_spec(retrieval_generator, wide=args.wide))
            query = key_of(context_entry(encoder, stranger,
                                         retrieval_generator))
            key_out.append(float((keys @ query).max()))

        retrieval[size] = {
            "accuracy": round(hits / size, 4),
            "key_accuracy": round(key_hits / size, 4),
            "key_then_verify_accuracy": round(verify_hits / size, 4),
            "key_plant_passes": 0,
            "key_then_verify_plant_passes": topk,
            "key_in_bank_similarity": round(sum(key_in) / len(key_in), 4),
            "key_outside_similarity": round(sum(key_out) / len(key_out), 4),
            "forward_passes_per_retrieval": size,
            "mean_margin_over_runner_up": round(sum(margins) / len(margins), 4),
            "in_bank_best_score": round(sum(in_bank_best) / len(in_bank_best), 4),
            "outside_bank_best_score": round(sum(outside) / len(outside), 4)}
    report["retrieval"] = retrieval

names = list(held)
report["wrong_context"] = {
    name: accuracy(plant, held[name], entries[names[(index + 1) % len(names)]])
    for index, name in enumerate(names)}

# Retention is not a question here -- entries are separate tensors and
# every weight is frozen -- but it is measured anyway, because F75's
# first version reported it in a form that could not have detected a
# leak.
report["retention_delta"] = {
    name: round(accuracy(plant, held[name], entries[name])
                - report["held_out"][name]["read_accuracy"], 4)
    for name in held}

print(json.dumps(report, indent=2))
if args.json:
    with open(args.json, "w") as handle:
        json.dump(report, handle, indent=2)

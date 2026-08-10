"""Schema family: four task families whose DYNAMICS GENUINELY DIFFER.

F67-F70 measured a downward acquisition-cost curve: a stored transition
model makes each next task cheaper, where a stored policy makes each
next task dearer. But every rung of that ladder NESTED -- r2 is a line,
r3 an open grid, r4 a walled grid, so a model of r4's dynamics literally
contains r2's. A downward curve is exactly what nesting predicts, with
no generalisation involved at all. The instrument could not tell
compounding from nesting.

This probe is the instrument that can. Four families share NO surface:

  line    position on a bounded line; actions +/-1, clipped at the ends
  dial    three counters mod 8; actions increment/decrement one, wrapping
  toggle  six bits; actions XOR a fixed mask -- self-inverse, abelian,
          nothing "moves" and there is no notion of adjacency
  perm    an ordering of four items; actions swap adjacent positions --
          non-abelian, state IS the arrangement

No family nests in another. A perfect model of `line` says nothing
whatever about which bits `toggle` flips. What they DO share is schema:

  * state is FACTORED into slots, each holding one value
  * an action changes a SMALL LOCAL subset of slots
  * unchanged slots are COPIED forward
  * effects COMPOSE, so a path is a sequence of actions
  * distance-to-goal is the length of the shortest such path

That last group is the only thing available to transfer, and it is the
abstract claim the project rests on: that seemingly unrelated tasks have
common structure a controller can learn top-down. If a model trained on
three families makes the fourth cheaper than cold, the structure is
real and the controller found it. If cost falls back to cold, then
F67-F70 measured nesting and are scoped to nested families only.

Prediction recorded BEFORE the run (F70): the shared schema is mostly
the copy-forward default -- "an action leaves most of the state alone".
A cold model must rediscover it every time; a sequential model has it
already. So the expected signature is a large saving on the FIRST
portion of each task's learning and a residual cost for the family's own
content, i.e. a real but PARTIAL discount, not the near-zero of nesting.

Controls, because eight measurement bugs in this project were caught by
controls and none by inspection:

  no-agent   random actions, per family, its own measured floor
  scramble   same state spaces, same action counts, but each family's
             dynamics replaced by a RANDOM PERMUTATION of its states.
             Sizes and interfaces identical; schema destroyed. If the
             sequential arm still beats cold here, the gain is generic
             network warm-up and the schema story is wrong.
  cold       every family also trained from scratch, so each has its own
             baseline rather than being compared to a single number.

Cost is updates ACTUALLY SPENT to predict a family's dynamics at
`--stop-at`, not a flat allowance -- F68's caveat. Behaviour is derived
by breadth-first search in the learned model, never by a stored policy.
Because every state space here is small, the model's entire predicted
transition table is materialised in ONE batched forward and the search
runs over that table, so search cost cannot contaminate the comparison.
"""

from __future__ import annotations

import argparse
import json
from collections import deque

import torch

from experiments.games_amodal.probes.schema_families import (
    ACTIONS, FAMILIES, SLOTS, VALUES, WIDTH, DenseModel, Family,
    SlotModel)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=69316)
parser.add_argument("--order", default="line,dial,toggle,perm",
                    help="sequence the warm arm learns, comma separated")
parser.add_argument("--updates", type=int, default=600,
                    help="per-family budget cap; cost is what is SPENT")
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--stop-at", type=float, default=0.98,
                    help="dynamics accuracy that counts as learned")
parser.add_argument("--depth", type=int, default=7, help="search depth")
parser.add_argument("--max-distance", type=int, default=6,
                    help="goals are sampled within this true distance, so "
                         "a depth-limited search can reach them at all -- "
                         "unreachable targets contaminated an earlier "
                         "measurement with sentinel distances")
parser.add_argument("--trials", type=int, default=32)
parser.add_argument("--steps", type=int, default=0,
                    help="step budget; 0 means max-distance + 2. The first "
                         "version allowed 24 steps in state spaces of 8 to "
                         "512, so a RANDOM WALK reached the goal 0.729 of "
                         "the time on `line` and the metric was mostly "
                         "measuring the step budget. A budget just above "
                         "optimal is what makes reach informative.")
parser.add_argument("--hidden", type=int, default=128)
parser.add_argument("--dim", type=int, default=64, help="slot-model width")
parser.add_argument("--lr", type=float, default=3e-3)
parser.add_argument(
    "--arch", choices=("dense", "slot"), default="dense",
    help="`dense` is the F71/F72 baseline: a flat MLP where each slot "
         "owns private weights, so a rule learned at slot 0 says nothing "
         "about slot 5. `slot` shares weights ACROSS slots -- one value "
         "embedding, one per-slot MLP, one output head, applied to every "
         "slot, with positional embeddings keeping slots distinct and "
         "attention supplying cross-slot interaction. F72's diagnosis "
         "was that the dense model learns six unrelated per-slot "
         "mappings and never the copy-forward rule they share; this is "
         "the architectural fix that makes structure shared by "
         "construction and leaves only content to be learned per family.")
parser.add_argument("--scramble", action="store_true",
                    help="control: replace every family's dynamics with a "
                         "random permutation of its own states")
parser.add_argument("--json", default="")
args = parser.parse_args()

torch.manual_seed(args.seed)
STEPS = args.steps or (args.max_distance + 2)


def make_model() -> tuple:
    net = (SlotModel(args.dim) if args.arch == "slot"
           else DenseModel(args.hidden))
    return net, torch.optim.Adam(net.parameters(), lr=args.lr)


def slot_loss(logits: torch.Tensor, targets: torch.Tensor,
              slots: int) -> torch.Tensor:
    blocks = logits
    total = 0.0
    for slot in range(slots):
        total = total + torch.nn.functional.cross_entropy(
            blocks[:, slot, :], targets[:, slot])
    return total / slots


def exact_accuracy(net, family: Family) -> float:
    """Exhaustive over every (state, action) pair -- no sampling noise."""
    size = len(family.states)
    states = torch.arange(size).repeat_interleave(family.actions)
    acts = torch.arange(family.actions).repeat(size)
    with torch.no_grad():
        logits = net(family.slot_values(states), acts)
    blocks = logits.argmax(-1)
    nxt = torch.tensor([family.table[int(s)][int(a)]
                        for s, a in zip(states, acts)])
    want = family.slot_targets(nxt)
    correct = torch.ones(states.shape[0], dtype=torch.bool)
    for slot in range(family.slots):
        correct &= blocks[:, slot] == want[:, slot]
    return round(float(correct.float().mean()), 4)


def slot_accuracy(net, family: Family) -> float:
    """Fraction of INDIVIDUAL slots predicted right, over every (s, a).

    Exact-match accuracy is all-or-nothing and hides partial structure.
    The schema on offer here is mostly copy-forward -- 'an action leaves
    most of the state alone' -- which shows up as slot accuracy well
    above the copy baseline while exact accuracy is still at chance.
    Without this readout a real but partial transfer is invisible.
    """
    size = len(family.states)
    states = torch.arange(size).repeat_interleave(family.actions)
    acts = torch.arange(family.actions).repeat(size)
    with torch.no_grad():
        logits = net(family.slot_values(states), acts)
    blocks = logits.argmax(-1)
    nxt = torch.tensor([family.table[int(s)][int(a)]
                        for s, a in zip(states, acts)])
    want = family.slot_targets(nxt)
    hits = sum(float((blocks[:, slot] == want[:, slot]).float().mean())
               for slot in range(family.slots))
    return round(hits / family.slots, 4)


def copy_baseline(family: Family) -> float:
    """Slot accuracy of the trivial rule 'next state = current state'.

    This is the number slot accuracy must BEAT to mean anything. If a
    transferred model only matches this, it has learned to sit still,
    not to predict.
    """
    size = len(family.states)
    states = torch.arange(size).repeat_interleave(family.actions)
    acts = torch.arange(family.actions).repeat(size)
    nxt = torch.tensor([family.table[int(s)][int(a)]
                        for s, a in zip(states, acts)])
    here, want = family.slot_targets(states), family.slot_targets(nxt)
    hits = sum(float((here[:, slot] == want[:, slot]).float().mean())
               for slot in range(family.slots))
    return round(hits / family.slots, 4)


def learn(net, opt, family: Family, updates: int, stop_at: float) -> tuple:
    """Self-supervised dynamics from random play. Returns (acc, spent)."""
    generator = torch.Generator().manual_seed(args.seed + 31)
    size = len(family.states)
    accuracy = exact_accuracy(net, family)
    if accuracy >= stop_at:
        return accuracy, 0
    for step in range(updates):
        states = torch.randint(0, size, (args.batch_size,),
                               generator=generator)
        acts = torch.randint(0, family.actions, (args.batch_size,),
                             generator=generator)
        nxt = torch.tensor([family.table[int(s)][int(a)]
                            for s, a in zip(states, acts)])
        logits = net(family.slot_values(states), acts)
        loss = slot_loss(logits, family.slot_targets(nxt), family.slots)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % 25 == 0:
            accuracy = exact_accuracy(net, family)
            if accuracy >= stop_at:
                return accuracy, step + 1
    return exact_accuracy(net, family), updates


def predicted_table(net, family: Family) -> list:
    """The model's WHOLE transition table, in one batched forward, so the
    search below costs nothing and cannot contaminate the cost figures."""
    size = len(family.states)
    states = torch.arange(size).repeat_interleave(family.actions)
    acts = torch.arange(family.actions).repeat(size)
    with torch.no_grad():
        logits = net(family.slot_values(states), acts)
    blocks = logits.argmax(-1)
    table = [[0] * family.actions for _ in range(size)]
    for row in range(states.shape[0]):
        predicted = tuple(int(blocks[row, slot])
                          for slot in range(family.slots))
        table[int(states[row])][int(acts[row])] = family.index.get(
            predicted, int(states[row]))
    return table


def search_action(table: list, start: int, goal: int, actions: int) -> int:
    """Breadth-first search IN THE LEARNED TABLE. No policy involved."""
    queue, seen = deque([(start, None)]), {start}
    while queue:
        current, first = queue.popleft()
        for action in range(actions):
            successor = table[current][action]
            head = action if first is None else first
            if successor == goal:
                return head
            if successor not in seen:
                seen.add(successor)
                queue.append((successor, head))
    return 0


def trial_pairs(family: Family) -> list:
    """Start/goal pairs at a true distance the search can actually cover."""
    generator = torch.Generator().manual_seed(args.seed + 900)
    size = len(family.states)
    pairs = []
    guard = 0
    while len(pairs) < args.trials and guard < args.trials * 200:
        guard += 1
        goal = int(torch.randint(0, size, (1,), generator=generator))
        field = family.distances(goal)
        options = [s for s, d in field.items()
                   if 1 <= d <= args.max_distance]
        if not options:
            continue
        pick = int(torch.randint(0, len(options), (1,), generator=generator))
        pairs.append((options[pick], goal, field[options[pick]]))
    return pairs


def reach(net, family: Family, *, random_actions: bool = False) -> dict:
    table = None if random_actions else predicted_table(net, family)
    generator = torch.Generator().manual_seed(args.seed + 1300)
    hits, ratios = 0, []
    pairs = trial_pairs(family)
    for start, goal, optimal in pairs:
        position, used = start, STEPS
        for step in range(STEPS):
            if position == goal:
                used = step
                break
            if random_actions:
                action = int(torch.randint(0, family.actions, (1,),
                                           generator=generator))
            else:
                action = search_action(table, position, goal, family.actions)
            position = family.table[position][action]
        if position == goal:
            hits += 1
            ratios.append(max(used, 1) / max(optimal, 1))
    return {"reach": round(hits / max(len(pairs), 1), 4),
            "path_ratio": round(sum(ratios) / len(ratios), 4) if ratios
            else None}


# ------------------------------------------------------------------- arms

order = [name.strip() for name in args.order.split(",") if name.strip()]
families = {name: Family(name, args.scramble, args.seed)
            for name in order}

report = {"seed": args.seed, "order": order, "scramble": args.scramble,
          "stop_at": args.stop_at, "budget": args.updates}

report["floors"] = {
    name: reach(None, family, random_actions=True)
    for name, family in families.items()}
report["copy_baseline"] = {name: copy_baseline(family)
                           for name, family in families.items()}
report["steps"] = STEPS

# Arm 1 -- COLD. A fresh model per family: its own honest baseline.
cold = []
for name in order:
    net, opt = make_model()
    # The untrained control for the warm arm's `slots_before`: whatever a
    # fresh network scores by initialisation alone. Transfer has to beat
    # THIS, not zero.
    slots_fresh = slot_accuracy(net, families[name])
    accuracy, spent = learn(net, opt, families[name], args.updates,
                            args.stop_at)
    cold.append({"family": name, "cost": spent, "accuracy": accuracy,
                 "slots_fresh": slots_fresh, **reach(net, families[name])})
report["cold"] = cold

# Arm 2 -- SEQUENTIAL. One model, families in order, each charged only
# what it actually spends. Zero-shot is measured BEFORE that family's
# training, so a family the prior already covers costs nothing.
net, opt = make_model()
warm = []
for name in order:
    family = families[name]
    before = exact_accuracy(net, family)
    slots_before = slot_accuracy(net, family)
    zero_shot = reach(net, family)
    accuracy, spent = learn(net, opt, family, args.updates, args.stop_at)
    warm.append({"family": name, "cost": spent, "accuracy": accuracy,
                 "accuracy_before": before, "slots_before": slots_before,
                 "zero_shot": zero_shot["reach"], **reach(net, family)})
report["warm"] = warm

# Retention: after the whole sequence, is every earlier family still
# predicted? A model cannot hold a contradiction, so this should be flat
# -- the claim that separates a model from a policy (F68).
report["retention"] = {name: exact_accuracy(net, families[name])
                       for name in order}
report["retention_reach"] = {name: reach(net, families[name])["reach"]
                             for name in order}

report["totals"] = {
    "cold": sum(entry["cost"] for entry in cold),
    "warm": sum(entry["cost"] for entry in warm)}

print(json.dumps(report, indent=2))
if args.json:
    with open(args.json, "w") as handle:
        json.dump(report, handle, indent=2)

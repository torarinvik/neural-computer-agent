# The library pays on tasks it has never seen (2026-08-15)

Status: **diagnostic**, on the already-consumed development seed, three
replicates over three different worlds. Nothing admitted;
`AgentBrain.bank` unchanged at `07319eb1`.

The integrated agent's library pays 2.79x and **every bit of it comes from
exact repeats**. That makes the store a cache. The claim a library is supposed
to support is much stronger: capability N+1 is cheaper because it is *built
out of* capabilities 1..N, on a task that has never occurred.

## Result

Four primitives, then eight composites the agent has never met, each a product
of two primitives under a boolean combiner. Twenty-four composites across three
replicates. **No task is clearable by a constant policy**, which is checked and
reported rather than assumed.

| arm | solved | composed | acquisition | vs its control |
| --- | :--: | ---: | ---: | ---: |
| **composing** | 24/24 | **23** | **928** | **0.288** |
| recognising (retrieval only) | 24/24 | 0 | 2976 | 0.930 |
| control (no library) | 24/24 | 0 | 3200 | 1.0 |
| **disjoint** (parts absent) | 24/24 | 5 | 3904 | **0.952** |
| shuffled feedback | **0/24** | 0 | 10752 | -- |

**A composite the agent has never seen costs 0.29 of what a fresh agent pays**
and 0.31 of what a library that can only retrieve pays -- worst replicate 0.33
and 0.35. And when the parts are missing the same mechanism gives **0.95**,
which is nothing.

It is not pattern-matching its way there. Of 24 composites it recovered the
**operator the task was actually built with 23 times**, and the exact pair of
parts 15 times. The other nine found a different decomposition that also works,
which is honest: a product of two machines frequently has more than one.

## Hierarchy: it builds on what it built

A composite the agent worked out and admitted is a record like any other, so it
can be a *part*. Nothing in the search knows about depth. Twenty triples across
five replicates:

| | solved | composed | acquisition |
| --- | :--: | ---: | ---: |
| **pairs seen first** | 20/20 | **17** | **896** |
| pairs withheld | 20/20 | 9 | 2112 |
| control | 20/20 | 0 | 3648 |

**0.246 of control.** Removing the intermediate layer, and nothing else, costs
0.58 on average -- so roughly half the advantage is the depth itself rather
than the primitives. That average hides a real spread: per replicate the depth
ratio is 0.31, 0.33, 0.40, 0.63 and **1.25**. In one world of five, having
built the pairs made the triples *more* expensive.

## The mechanism, and why it is cheap

Each stored program is executed once against the evidence in hand, producing a
press vector. Every pair under every combiner is then an elementwise boolean
operation on two cached vectors. A twenty-four record library offers **852
hypotheses** and not one costs an episode, a program execution, or a search.

A found combination is then materialised as the product Mealy machine of its
parts and compiled like any induced hypothesis, so what gets confirmed and
admitted is a real artifact rather than a promise about two other files.

## The correction that turned out to be wrong

Scoring is free, so the search is enormous, so the obvious move is to divide
alpha by the number of hypotheses examined. That was the first version. It is
the wrong trade, and the measurements are what say so.

Over **800 unrelated targets** against a 24-record library:

| | wrong composites adopted |
| --- | ---: |
| corrected | **0** |
| uncorrected | 5 |

**But all five were refused by confirmation.** Their true held-out accuracies
were 0.775, 0.775, 0.782, 0.792 and 0.798, against a gate of 0.8. Meanwhile the
correction costs six of eight composable tasks at 10% label noise, and at the
agent level it costs a factor of 1.8 in evidence (1664 against 928) while
composing exactly one more task.

Confirmation is not optional and runs on every candidate anyway. Tightening the
free gate until it can do the expensive gate's job buys a redundancy already
present and pays for it in noise tolerance. The correction is off by default and
kept for measurement.

## Adversarial checks

**Scale.** Libraries of 4, 8, 16 and 24 records, up to 852 hypotheses per task.
False adoptions do not grow with the library.

**Targets that cannot be built.** 800 unrelated rules. In 89 of them some wrong
pair clears the 0.8 floor -- up to **0.906** -- so the floor alone is not what
protects the search. What protects it is that on a buildable task a correct
candidate sits at 1.000 and outranks them, and on an unbuildable one
confirmation refuses what slips through.

**An operator the search does not have.** Composites built with implication,
which is not in the vocabulary. No wrong adoption at any library size or noise
level.

**Unreliable feedback.** At 10% label noise composition still fires on 18 of 24
composites, at 0.73 of the retrieval-only arm. Degraded, not collapsed.

**Destroyed feedback.** Permuting each probe's labels: **0/24 solved**, nothing
admitted, and the full ladder spent on every task.

**Its own control per stream.** The disjoint arm runs a different stream, so it
is read against a control on *that* stream rather than on the main one -- a
comparison the first version got wrong by a factor that flattered it.

## What is honestly weak

**Composition can cost more.** The disjoint arm ran to 1.04 of its control in
one replicate, and the hierarchical depth ratio reached 1.25 in one of five. A
composite that is adopted and fails confirmation costs two full episodes, and
nothing recovers that.

**The search vocabulary matches the world's.** Composites are products under
and/or/xor and the search tries and/or/xor. That is a fair test of whether the
parts can be used and a generous one about knowing how they combine. A world
composing some other way is only covered by the implication probe, which tests
that it *refuses*, not that it adapts.

**Pairs only.** Depth comes from admitted composites becoming parts, not from
searching triples directly, so a three-part task is only reachable if a
two-part one was met first. In one replicate that route was not taken.

**Nothing is admitted here.** These runs use scratch libraries on a spent
development seed. The holdout that admitted programs is
`brainworkshop_integrated_agent_2026-08-15`, and composition is not in it.

**Still one alphabet, one frontend, one modality**, and the composites are
Mealy machines like everything else in this session.

## Reproducing

```bash
PYTHONPATH=src .venv/bin/python -m experiments.brainworkshop_canonical.compositional_transfer
```

About ninety seconds.

```bash
PYTHONPATH=src .venv/bin/python -m experiments.brainworkshop_canonical.hierarchical_transfer
```

```bash
PYTHONPATH=src .venv/bin/python -m experiments.brainworkshop_canonical.composition_adversarial
```

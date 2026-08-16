# Every piece at once, with nothing handed over (2026-08-16)

Status: **diagnostic**, on the already-consumed development seed. Nothing
admitted; `AgentBrain.bank` unchanged at `07319eb1`.

Five records, five isolated axes. Two of them oracle identification outright;
three take the decomposition as given. Each was right to change one thing at a
time, and between them they say nothing about where the system actually is.

This runs all of it together on the hardest goal language -- relations to a
target that moves -- and the number it is after is the **compounding**.

## Result

Four worlds, twenty-step episodes, six relations, three of which were never a
reward. Normalised so the worst achievable is 0 and exact finite-horizon
optimal is 1.

| | integrated | told_all | random | identification right |
| --- | ---: | ---: | ---: | ---: |
| trained relations | **0.598** | 0.865 | 0.310 | 0.542 |
| held-out relations | **0.610** | 0.773 | 0.265 | 0.515 |

**Taking away the oracles costs about a quarter of optimal**, with the self
model in place: 0.865 falls to 0.598 and 0.773 to 0.610. Without it the same
agent scores 0.557 and 0.537, so a third of the gap is closed by remembering
who it was last episode and the rest is still open. The integrated agent is
roughly twice acting-blindly either way -- it works, and the per-axis records
overstate it.

**Selecting the cut is free.** Description length chose components in every
task of every world, which is what the decomposition record predicted, so the
whole of the shortfall is identification.

**The agent knows which marker it is about half the time**, online, and the
next section is about why that is the number to look at rather than the 0.92 it
reaches offline.

The model it builds is only slightly worse than the oracled one -- coverage
0.954 against 0.978 -- so the damage is not mainly in what it learns. It is in
not knowing, at each step, which configuration it is in.

> An earlier version of this run let the exploration policy consult the oracle
> to decide which action was untried, which chose the integrated arm's
> trajectory for it. Fixed: the policy is keyed on the raw reading, which needs
> no identification. That moved every arm, `told_all` included, from 0.963 to
> 0.865 on trained relations -- so the numbers here are lower than the ones in
> the sweep tables below, which were taken before the fix and are kept as
> relative comparisons rather than absolutes.

## Identity as a persistent cause: the one thing that worked

Every mechanism up to here re-derived "which marker am I" from a single
episode's evidence, while the world stayed the same world across all forty.
Carrying a **self model** across episodes -- my place, under my action -- and
re-fitting it by alternating "which track was me" with "what do I do" over
frames already collected changes that. No new experience is spent; it is
arithmetic over experience already paid for.

It only works **soft**. Both variants were run:

| variant | identification, by re-fitting pass |
| --- | --- |
| hard: name a track, learn its dynamics | 0.47 0.47 0.47 0.47 0.47 0.47 |
| | 0.42 0.42 0.42 0.42 0.42 0.42 |
| | 0.53 0.53 0.53 0.53 0.53 0.53 |
| **soft: weight by posterior** | **0.88 0.93 0.95 0.95 0.95 0.95** |
| | **0.78 0.82 0.85 0.85 0.85 0.88** |
| | **0.90 0.90 0.93 0.93 0.93 0.93** |

**The hard loop never moves.** It reaches its fixed point in one pass, because
a model learned from a wrong naming re-confirms that naming -- a self-confirming
identity loop. Weighting each episode by a likelihood-derived posterior lets an
ambiguous episode contribute almost nothing instead of a confident mistake, and
identification climbs from **0.47 to 0.92**.

Thresholded agreement was replaced by likelihood for the same reason: a track
seen twice and agreeing twice is not the claim a track seen twenty times and
agreeing eighteen makes, and a ratio cannot tell them apart.

### And it only partly reaches the behaviour

| self model | trained | held out | identification, online |
| --- | ---: | ---: | ---: |
| none | 0.557 | 0.537 | 0.481 |
| hard | 0.612 | 0.536 | 0.547 |
| **soft posterior** | 0.598 | **0.610** | 0.542 |

Held-out return improves by 0.073, about a quarter of the remaining distance to
the oracled arm. But **online identification moves only 0.48 to 0.54**, against
0.92 offline, and that gap is the honest headline. The self model is learned
well; applying it inside a *fresh* episode still needs enough within-episode
history to score the tracks against, and the early steps of an episode do not
have it. Knowing what I do is not the same as knowing, right now, which of
these two markers is doing it.

**This is a development-seed result and is not held out.** The holdout block
was spent on the five component records. A fresh block, with the success
criterion written before the run, is what this needs before it counts.

## Two obvious fixes, both measured, both worse

The diagnosis suggests waiting for better evidence before committing, and
revising the decision as more arrives. Both were implemented and swept:

| minimum contrasts | re-decide each step | trained | held out | orientation |
| ---: | --- | ---: | ---: | ---: |
| **0** | **no** | **0.644** | 0.520 | **3.6** |
| 0 | yes | 0.573 | **0.535** | 3.6 |
| 2 | no | 0.534 | 0.440 | 6.0 |
| 2 | yes | 0.527 | 0.418 | 6.0 |
| 6 | no | 0.491 | 0.382 | 10.8 |
| 6 | yes | 0.458 | 0.394 | 10.8 |

**Requiring more evidence degrades the agent monotonically.** The orientation
delay it buys -- 3.6 steps to 10.8, more than half the episode -- costs far
more than the better identification is worth. Re-deciding every step is a wash:
worse on trained relations, marginally better on held out.

The simplest configuration is the best one, which is not the result the fix was
written to produce. It is recorded because the alternative is to keep the
knobs and quietly report only the setting that won.

## Why identification does not work here, precisely

Traced on a clean synthetic stream -- agent on a deterministic table,
distractor on a fixed circuit, both moving every step -- the correspondence
beam follows the agent for four steps and then **swaps onto the distractor**.
Both tracks end up mixtures, matching the agent 6 and 7 frames of 12
respectively, and `identify_roles` names whichever mixture scores higher.

This is the same limit the holdout found from the other side: on unseen
worlds, searching over correspondences does no better at naming the agent than
the naive elimination rule. Two identical markers, both moving, in a world
where an action teleports you, leave the correspondence genuinely
underdetermined over a short history. The beam recovers it at 24 steps and
does not at 12.

A test now asserts the wrong answer is reachable, so that if this ever becomes
reliable the record is flagged as stale rather than silently outgrown.

## What is honestly weak

**One development seed, four worlds.** The holdout block was spent on the five
component records; this has not been held out and should be.

**Identification is the only oracle removed that matters**, so this measures
one compounding rather than four. Cut selection is free and exploration and the
goal language were never oracled.

**Designing the probe does not work, and four attempts say so.** Choosing the
action whose outcome most separates the surviving correspondences scores 0.640
against random's 0.644; adding a term for unpredictable outcomes drops it to
0.573; seeking untried actions alone gives 0.577; and seeking *matched
contrasts* -- the thing controllability is actually measured from -- gives
0.642. Random probing is already near-optimal, because a matched contrast needs
the same place left by two different actions, and every systematic policy
produces fewer of those than chance does. The bottleneck was never the policy.

**`told_all` is itself a development-seed number** from the relational record,
whose held-out sub-claim the holdout weakened. The 0.3 gap is against a
ceiling that is itself approximate.

## Reproducing

```bash
PYTHONPATH=src .venv/bin/python -m experiments.brainworkshop_canonical.integrated_navigation --tasks 4
```

About a minute.

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_integrated_navigation.py -q
```

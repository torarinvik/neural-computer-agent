# The five navigation results, on worlds nobody had looked at (2026-08-16)

Status: **holdout**, three replicates on block `navigation_family_holdout`
(7000017, 7500017, 8000017), asserted unused before any episode was drawn.
`AgentBrain.bank` unchanged at `07319eb1`. Nothing admitted.

## A sharper problem than the usual one

All five navigation experiments drew their worlds from a **hard-coded** seed,
`9000 + 37 * index`. Changing the run seed varied only the exploration
randomness -- the sampled worlds were the same worlds every time. A rerun at a
new seed would have looked like a holdout and been nothing of the kind, and
every one of those five records would have been "confirmed" against exactly the
worlds it was developed on.

The world seed is now a parameter, defaulting to 9000 so every recorded
diagnostic still reproduces byte-for-byte, and this run passes seeds from a
block that has never been spent.

## Method

The five records made **43 checkable orderings** between them -- including both
negative results, which are held to the same standard. Each is extracted as a
boolean and counted across three replicates. Nothing was tuned: every
threshold, arm, relation and candidate cut is whatever the development records
left it at.

## Result: 38 of 43 held in all three replicates

| family | held everywhere |
| --- | --- |
| successor transfer | **11 / 11** |
| learned decomposition | **5 / 5** |
| curious exploration | **7 / 7** |
| relational transfer | 5 / 6 |
| object identity | 8 / 12 |

Three of the five come through untouched, and the headline numbers move in the
agent's favour rather than against it:

| | development | holdout (3 worlds) |
| --- | ---: | ---: |
| successor, single-goal gpi | 0.801 | **0.895** |
| successor, accumulation 1 -> 4 policies | 0.728 -> 0.844 | **0.810 -> 0.940** |
| decomposition, components vs whole (bits) | 134.6 vs 359.8 | **128.2 vs 329.4** |
| exploration, uniform / optimistic / curious | 0.802 / 0.922 / 0.984 | **0.844 / 0.938 / 1.000** |
| exploration, ungated under a distractor | 0.974 -> 0.938 | **0.987 -> 0.927** |
| relational, place control (held out) | 0.296 | **0.336** vs 0.296 random |

Components was chosen in every task of every replicate. Gated curiosity was
bit-identical across the distractor conditions in every replicate; ungated was
not, in every replicate. Both negative results -- the degenerate horizon family
and optimism doing most of the work -- reproduced exactly.

## What did not survive

### The object identity headline was a development-seed artifact

The record claimed searching over correspondences finds the agent **1.3x more
often** than elimination. On unseen worlds it does not:

| condition | search | persistence | development claim |
| --- | ---: | ---: | --- |
| random_walk | 0.578 | 0.557 | 0.617 vs 0.445 |
| cycling | 0.609 | **0.615** | 0.625 vs 0.523 |

Search is unchanged; **elimination got luckier**. Persistence even wins the
cycling condition, and on world 7500017 search loses the two-marker *control*
(0.984 against 1.000). Four identity claims are withdrawn.

### But the thing that matters survived, and it is a better result

Identification accuracy is measured on the final frame. The **model** is what
anything downstream actually uses, and there the ordering is not close:

| condition | hybrid model | persistence model | alignment model |
| --- | ---: | ---: | ---: |
| random_walk | **0.667** | 0.484 | 0.517 |
| cycling | **0.693** | 0.492 | 0.508 |

Held in **all six cells** (three replicates x two distractor conditions),
range 0.625-0.758 against 0.453-0.539. The reason is now clear and was not
obvious before: a coherent track feeds *coherent transitions* to the model even
on episodes where the final naming comes out wrong, whereas elimination that
happens to be right on the last frame has been feeding a mixture of two objects
all the way through. Two claims were added for this after the fact, and are
marked as such in the code.

Track fidelity also held everywhere: search 0.665-0.725 against greedy
alignment 0.583-0.586, and greedy min-change is reliably the worst arm.

### The PGM cost is real on average and unreliable per world

The relational record reported held-out relations costing **0.16** of optimal
against re-solving. Per replicate:

| world | gpi | replan | gap |
| --- | ---: | ---: | ---: |
| 7000017 | 0.868 | 0.865 | **-0.002** |
| 7500017 | 0.808 | 0.857 | +0.049 |
| 8000017 | 0.778 | 0.972 | +0.194 |

Mean +0.080, and it clears the record's 0.05 bar in one world of three. The
finding is directionally right and much weaker and noisier than reported. The
record said the gap "does not survive at two tasks"; the real explanation is
that it varies by *world*, which is a worse explanation than the one written.

Every other relational claim held in all three: pairs remain necessary (place
control 0.336 against 0.296 random, versus 0.818 for pairs), and a relation
never rewarded still transfers at 0.818 against 0.446 for the best stored
policy.

## What this changes

1. **Two records are corrected in place**, not defended. The identity headline
   moves from identification accuracy to model accuracy; the relational record
   gets the per-world spread instead of a single number.
2. **The claim that survived was not the claim that was made.** Search buys a
   coherent *track*, not a correct *name*, and a coherent track is worth 0.18
   of model accuracy. That is a mechanism-level correction, and it only showed
   up because the holdout scored more than the headline.
3. **Successor features, decomposition and curiosity need no correction.**
   Twenty-three claims across three families, three unseen worlds, no failures.

## What is honestly weak

**Three worlds per replicate, four tasks each.** Trimmed from the development
counts to keep three replicates inside the ten-minute rung.

**The claims are the ones the records happened to state.** A record that
phrased a finding loosely is checked loosely. Two claims were added *after*
seeing the data, which is exactly the move this run exists to catch --they are
flagged in the code and should be treated as development-seed claims until a
future block tests them.

**Nothing here is a fresh mechanism.** It is the same code on unspent
experience, which is the only thing it was supposed to be.

## Reproducing

```bash
PYTHONPATH=src .venv/bin/python -m experiments.brainworkshop_canonical.navigation_holdout --tasks 4 --replicates 3
```

About thirty-four minutes.

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_navigation_holdout.py -q
```

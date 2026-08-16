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

| | integrated | told_all | random | orientation | identification right |
| --- | ---: | ---: | ---: | ---: | ---: |
| trained relations | **0.644** | 0.963 | 0.310 | 3.6 / 20 | 0.524 |
| held-out relations | **0.520** | 0.806 | 0.265 | 3.6 / 20 | 0.471 |

**Taking away the oracles costs about a third of optimal.** 0.963 falls to
0.644, and 0.806 falls to 0.520. The integrated agent is still twice
acting-blindly, so it works -- but the per-axis records overstate it by
roughly 0.3, and that gap is the honest position of the system.

**Selecting the cut is free.** Description length chose components in every
task of every world, which is what the decomposition record predicted, so the
whole of the shortfall is identification.

**The agent knows which marker it is about half the time.** 0.524 and 0.471 of
scored steps. It also spends 3.6 of 20 steps oriented at nothing, acting at
random while the correspondence search gathers enough history to say anything,
and those steps are charged rather than run off the clock.

The model it builds is only slightly worse than the oracled one -- coverage
0.954 against 0.978 -- so the damage is not mainly in what it learns. It is in
not knowing, at each step, which configuration it is in.

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

**The orientation protocol is crude**: act at random until the search says
something. A probe policy designed to *disambiguate* -- take the action whose
outcome most separates the candidate correspondences -- is the obvious thing
and is not tried here.

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

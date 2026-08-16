# Exploring on purpose, and the television that stops it (2026-08-16)

Status: **diagnostic**, on the already-consumed development seed. Nothing
admitted; `AgentBrain.bank` unchanged at `07319eb1`.

The caveat this closes, standing since the navigation work:

> **The probe policy is uniform and unexamined.**

Four mechanisms were taken from the curiosity and Agent57 literature. Two of
them work here, one is unnecessary by construction, and one is degenerate. All
four are reported.

## Novelty is a task, and that is why this was cheap

Successor features store what a policy goes on to *see*, not what that is
worth. Wanting the least-visited place is then a weight vector -- and it
changes every single step, which is exactly the case successor features make
free. Re-aiming the agent costs one dot product per stored policy. The
occupancies are refreshed once per episode, when the *model* changes; the task
changing every step costs nothing.

There is no intrinsic reward channel here, no bonus added to a return, and no
second value function. There is a `w`.

## Result

Six worlds, a budget of **60 steps** (6 episodes of 10), coverage measured over
the 32 `(place, action)` cells. Downstream is the held-out-goal evaluation from
the successor transfer record, run afterwards with no further experience.

| | | coverage | steps to 90% | reached | downstream |
| --- | --- | ---: | ---: | ---: | ---: |
| **no distractor** | uniform | 0.802 | 60.0 | 0/6 | 0.845 |
| | optimistic | 0.922 | 50.0 | 5/6 | 0.902 |
| | **curious** | **0.984** | **41.7** | **6/6** | **0.916** |
| | curious (ungated) | 0.974 | 41.7 | 6/6 | 0.916 |
| | curious (bandit) | 0.984 | 41.7 | 6/6 | 0.916 |
| **random-walk distractor** | uniform | 0.802 | 60.0 | 0/6 | 0.845 |
| | optimistic | 0.922 | 50.0 | 5/6 | 0.902 |
| | **curious** | **0.984** | **41.7** | **6/6** | **0.916** |
| | curious (ungated) | 0.938 | 51.7 | 6/6 | 0.909 |
| | curious (bandit) | 0.984 | 41.7 | 6/6 | 0.916 |

Acting blindly downstream scores 0.116, so every arm here is doing something.

**Directed novelty beats uniform wandering**: 0.984 against 0.802 coverage,
reaching 90% in 42 steps where uniform never reaches it inside the budget, and
0.916 against 0.845 downstream on goals it was never aimed at.

**Most of that is optimism, not curiosity.** Trying an action never tried here,
then wandering, gets 0.922 on its own. Of the 0.182 coverage gained over
uniform, optimism accounts for 0.120 and the novelty machinery for 0.062. The
`optimistic` arm exists so that this cannot be quietly folded into the
headline.

## The noisy television, measured

**Gated curiosity is bit-identical in both conditions** -- 0.984, 41.7 steps,
0.916 downstream, with and without something else moving in the frame. Novelty
counted only over the part of the scene the agent was measured to control
cannot see the distractor at all.

**Ungated curiosity is not**: 0.974 -> 0.938 coverage, 41.7 -> 51.7 steps,
0.916 -> 0.909 downstream. Counting novelty over whole readings works while the
reading is essentially the agent's own place; put a third marker in the frame
and no reading ever repeats, so a place stood on forty times still reads as
half-new.

The degradation is partial, and the reason is worth stating: a place never
visited has no readings at all and still stands out, so the agent still finds
the obvious gaps. What erodes is the contrast that separates two *visited*
places -- which is the part exploration needs late, when the obvious gaps are
gone. Measured directly: the ungated spread falls to 0.43 of its
no-distractor value while the gated vector is unchanged.

This is the first place the controllability measure from the identity record
has been used for something other than identification, and it is what makes
the gating possible.

## Two mechanisms that did not earn their place

**Agent57's split value heads are unnecessary here, structurally.** They split
Q into extrinsic and intrinsic parts because one shared function approximator
cannot hold two reward scales -- their ablation drops a game to random-policy
scores without it. That is a fact about representational interference, and psi
in cumulant space has none: value is linear in the task, so one stored
occupancy answers `w_e + beta * w_i` exactly, for every beta, with no second
store. Asserted as a test rather than run as an arm, because there is nothing
to measure.

**The family of horizons is degenerate, and so the meta-controller is too.**
Discounts 0.5, 0.95 and 0.99 give *identical* coverage, 0.984 in every case,
and the bandit therefore "matches the best fixed arm" only because every arm is
the same arm.

The cause is structural rather than a matter of this world being small: in a
deterministic world the greedy policy for "be at place p" is the shortest route
to p, and shortest is shortest at any discount in (0, 1). Verified at 8 of 8
base tasks. **A policy family that differs has to differ in its cumulants, not
in its horizon** -- which is what the Option Keyboard actually varies, and is
the forward pointer to relation-valued features.

The bandit is kept, working and tested, because the family it needs is a
change to what the base policies *are*, not to the controller that picks
between them.

## What is honestly weak

**Optimism does most of the work**, and on a world of 32 cells that is not
surprising. The novelty machinery should be re-measured where coverage is
genuinely hard -- more places, or regions reachable only by long routes -- and
until then its 0.062 is a small effect on an easy problem.

**Identification is handed over throughout** by the declared place-to-cluster
oracle, so that the distractor moves one axis and not two. The identity record
measures what identification costs; this does not.

**The distractor degrades the ungated arm and never captures it.** The classic
noisy television holds an agent still indefinitely. This one costs it 10 steps.
The finding is directional, and a source of surprise the agent could *sit and
watch* -- rather than one that merely blurs its counts -- is not tested here.

**Six worlds, one seed, no holdout.**

## Reproducing

```bash
PYTHONPATH=src .venv/bin/python -m experiments.brainworkshop_canonical.curious_exploration --tasks 6
```

About four minutes.

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_novelty.py tests/test_curious_exploration.py -q
```

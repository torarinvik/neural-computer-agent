# A task becomes a vector, and the model stops being thrown away (2026-08-16)

Status: **diagnostic**, on the already-consumed development seed. Nothing
admitted; `AgentBrain.bank` unchanged at `07319eb1`.

Two caveats from the object-navigation record, closed with one import.

> **Goals are places, not descriptions.** "Go where this marker is" is the whole
> goal language.

> **The model is still per-task and discarded.** It is not compiled, not
> admitted, and not composed.

Successor features (Barreto et al. 2017; PNAS 2020) answer both with the same
object. A policy stores the discounted future occupancy of *features* rather
than a value; a task is a weight vector over those features; the value of any
stored policy under any new task is one dot product,

    Q^pi_w(s, a) = psi^pi(s, a) . w

and generalised policy improvement acts greedily across everything stored.

**It is exact here, and gradient-free.** Every published version approximates
`psi` with a network. Eight places and four actions make it the solution of a
linear system, `(I - gamma P)^-1 P`, so nothing is trained and the number it
reports *is* the return -- checked against a rollout to 1e-7.

## The goal language, widened

`phi` is the indicator of the place arrived at, so a one-hot `w` reproduces the
previous experiment exactly. Everything else is a different vector over the
same features:

- `single` -- one target marker, **shown in the scene**;
- `disjunction` -- two target markers, "reach either", **shown**;
- `avoid` -- one target and a negative weight on a hazard, **given**.

The hazard is given rather than shown, and that is a real limit rather than a
convenience: one colour cannot say "not here", and a second colour would undo
the reason the markers share one. It is declared here and in the code.

## Result

Eight sampled worlds. Four policies induced for the training goals and stored;
then tasks over places never trained on, with **no further experience**.
Normalised so the worst achievable return is 0 and the best is 1, which is the
only scale that survives negative weights.

| | gpi | best stored | replan | random |
| --- | ---: | ---: | ---: | ---: |
| single | **0.801** | 0.356 | 0.951 | 0.203 |
| disjunction | **0.753** | 0.408 | 0.771 | 0.296 |
| avoid | **0.881** | 0.655 | 0.971 | 0.556 |

With identification handed over, so that only the choice of action is being
measured:

| | gpi | replan |
| --- | ---: | ---: |
| single | 0.843 | 0.999 |
| disjunction | 0.889 | 0.999 |
| avoid | 0.907 | 1.000 |

**A task the goal language could not previously express is solved at 75-88% of
what is achievable, with no planning and no new experience.** Following the
best single stored policy gets 36-66%; acting blindly gets 20-56%.

**And it is not free.** Re-solving the whole task from scratch does better --
0.95 against 0.80 on `single`, 0.97 against 0.88 on `avoid`. Generalised policy
improvement buys away the search and pays about a tenth of the achievable
return for it. That is the honest trade and it should not be reported as a win
over planning.

## The accumulation claim, measured directly

Offline, in the model, with no environment at all: how close does stitching get
to the re-solved optimum as the library grows?

| stored policies | gpi / optimal | best single / optimal |
| ---: | ---: | ---: |
| 1 | 0.728 | 0.067 |
| 2 | 0.765 | 0.077 |
| 3 | 0.788 | 0.108 |
| 4 | **0.844** | 0.142 |

**More stored policies means novel tasks are solved better, without any further
experience.** That is what the accumulation machinery was always for and it is
the first time it has been pointed at the navigation work.

One stored policy already reaches 0.728 because generalised policy improvement
over a single `psi` is still a one-step policy improvement over that policy --
which is why `best single` at 0.067 is the right floor to read it against.

The store is `SuccessorFeatureLibrary`: append-only, checksummed, digest over
records in order, and a *behavioural* duplicate test on what the policy does at
every place. Same discipline as `induced_library`, for the same reason.

## An error worth recording

The first version of `transfer_gap` compared `max over actions of (max over
policies)` against `max over policies of (max over actions)`. Those are the
same number -- the maximum of the matrix -- so it reported a gain of exactly
zero at every state, which is what a tautology looks like from outside. The
gain has to be the value of the **stitched policy**, evaluated in its own
right, against the value of the best stored one. There is now a test that
asserts the two maxima coincide, so nobody re-derives the same metric.

## What is honestly weak

**Beating the stored policies is trivial and is not the claim.** A policy built
for goal 3 earns nothing on goal 6, so the first measurement gave gains of 19
out of a possible 20 and meant nothing. Everything above is against the
re-solved optimum instead.

**The hazard is given, not shown.** One-third of the widened goal language
still arrives through a declared oracle rather than through pixels.

**Cumulants are hand-chosen.** `phi` is the place indicator because that is
what makes the previous experiment a special case. Nothing here discovers what
the features should be, and that is the whole of what makes the goal language
finite: no property, no relation, no conjunction over anything but places.

**One task, one library.** Policies accumulate across goals within a world and
are discarded between worlds. Nothing transfers across dynamics yet.

**Identification still costs.** 1.34 steps of 20 are spent working out which
marker is the agent, and the gap between the plain and `told` arms (0.04-0.14)
is what that is worth.

**Eight worlds, one seed, no holdout.**

## Reproducing

```bash
PYTHONPATH=src .venv/bin/python -m experiments.brainworkshop_canonical.successor_transfer --tasks 8
```

About six minutes.

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_successor_features.py tests/test_successor_transfer.py -q
```

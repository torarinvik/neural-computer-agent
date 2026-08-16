# Which one am I, when something else is moving too (2026-08-16)

Status: **diagnostic**, on the already-consumed development seed. Nothing
admitted; `AgentBrain.bank` unchanged at `07319eb1`.

The caveat this attacks, quoted from the object-navigation record:

> **Two objects, one static.** Identity is worked out from persistence, which
> works because the goal does not move. A second moving object, or a
> distractor, breaks that rule outright and nothing here proposes a
> replacement.

A third marker is added that moves for its own reasons. It changes neither the
dynamics nor the reward -- verified in a test -- so the only thing it breaks is
identification.

## The rule splits in half, and only one half breaks

The old rule was two rules. **The goal is the place in every observation**, and
**I am whatever is left**. A distractor leaves the first intact -- the goal is
still the only constant -- and destroys the second, because elimination now
leaves two candidates and takes whichever sorted first.

That is what the numbers show, and it is worth being precise about, because the
caveat as written blamed the wrong half.

## Result

Eight worlds, sixteen exploration episodes each, every arm reading **the same**
episodes.

| condition | arm | finds me | finds the goal | track fidelity | model correct |
| --- | --- | ---: | ---: | ---: | ---: |
| none | hybrid | **1.000** | 1.000 | 0.999 | 0.968 |
| none | search | 1.000 | 0.984 | 0.999 | 0.968 |
| none | alignment | 1.000 | 1.000 | 0.882 | 0.963 |
| none | persistence | 1.000 | 1.000 | 0.882 | 0.963 |
| none | predictability | 0.516 | 0.391 | 0.882 | 0.546 |
| random_walk | hybrid | **0.617** | **1.000** | 0.669 | **0.581** |
| random_walk | search | 0.617 | 0.727 | 0.669 | 0.581 |
| random_walk | alignment | 0.406 | 1.000 | 0.571 | 0.462 |
| random_walk | persistence | 0.445 | 1.000 | 0.571 | 0.525 |
| random_walk | predictability | 0.180 | 0.000 | 0.571 | 0.148 |
| cycling | hybrid | **0.625** | **1.000** | 0.710 | **0.684** |
| cycling | search | 0.625 | 0.750 | 0.710 | 0.684 |
| cycling | alignment | 0.391 | 1.000 | 0.579 | 0.480 |
| cycling | persistence | 0.523 | 1.000 | 0.579 | 0.531 |
| cycling | predictability | 0.180 | 0.000 | 0.579 | 0.126 |

`none` is the control: two markers, the previous experiment, where everything
except `predictability` must still be perfect or something was broken in
passing.

**Searching over correspondences finds the agent about 1.3x more often than
elimination** (0.617 vs 0.445; 0.625 vs 0.523) and builds a model that is
correspondingly less wrong. **Intersection remains the better way to find the
goal** (1.000 against 0.727), which is why the best arm is the one that uses
each mechanism on the half it suits.

**A wrong identification does its damage through the model.** The agent that
thinks it is the distractor learns the distractor's dynamics: model accuracy
drops from 0.968 to 0.581.

## Why the obvious measure is the wrong one

`predictability` is in the table to be beaten. It picks the track it can
predict best, which sounds like the same idea and is not: a distractor on a
fixed circuit is **perfectly** predictable and completely unresponsive. It
scores 0.180 -- below chance for three tracks -- and finds the goal essentially
never.

Getting the right measure took three attempts, and the two failures are as
informative as the fix.

**Accuracy** says a marker moving at *random* is perfectly predictable, because
a table keyed by `(symbol, action)` fitted on twenty steps sees almost every
key once.

**Description length** is not fooled by that, and is unusable anyway: with
twenty steps and thirty-two possible keys, conditioning on the action can never
repay the table it costs, so every track -- the agent included -- scored
negative. MDL was correct about the evidence and silent as a discriminator.

**A matched contrast** works. Among visits to the same place, compare what
followed under *different* actions against what followed under the *same*
action twice. The second is the noise floor; the difference isolates the
action's effect and needs a handful of repeated visits rather than a populated
table. Measured on one 19-step stream: agent +1.00, random walker 0.00, cycling
distractor 0.00, static 0.00.

## Two more things that had to be true

**The probe policy must be random.** Cycling the actions makes the action a
function of the step, so the agent looks exactly as unresponsive as a
distractor on a circuit. Under a deterministic probe, identification scored
*below chance*. Controllability is a question about variation and there is no
answer without any.

**Displacement is the wrong alignment prior for this world.** AlignNet matches
by minimum movement, which assumes things move a little between frames; here an
action sends the agent to an arbitrary place, so the nearest centroid is
frequently the other object. Replacing it with minimum *symbol* change -- one
action moves one thing -- plus AlignNet's actual contribution, aligning against
slot memory rather than the previous frame, is what took the two-marker case
from 0.882 to 0.999 track fidelity.

## What is honestly weak

**Greedy alignment does not merely degrade with a distractor, it is wrong.**
Minimum-change assumes one thing moves at a time; with two movers that premise
is false. It follows the agent 57% of the time and **does not improve with
eight times the experience** -- measured at 20, 40, 80 and 160 steps. That is a
wrong prior, not a data shortage, which is why search replaced it rather than
being given more episodes.

**Search recovers most of the gap, not all of it.** 0.617-0.625 against 1.000
in the two-marker case. Three identical teleporting markers leave genuine
ambiguity: every frame is a *set* of places, and several correspondences remain
consistent with everything observed.

**A static decoy is not addressed and is not addressable here.** A second
marker that never moves is indistinguishable from the goal by dynamics alone.
Nothing in this experiment claims otherwise, and no arm was given the chance to
fail at it.

**The beam is width 8 over at most six tracks.** It is exhaustive per step and
approximate over the episode, and nothing bounds how far the kept hypotheses
are from the best one.

**Identification is scored at the end of the episode**, against where the agent
actually was, rather than continuously.

**Eight worlds, one seed, no holdout.**

## Reproducing

```bash
PYTHONPATH=src .venv/bin/python -m experiments.brainworkshop_canonical.object_identity --tasks 8
```

About thirty-five seconds.

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_object_identity.py tests/test_slot_alignment.py -q
```

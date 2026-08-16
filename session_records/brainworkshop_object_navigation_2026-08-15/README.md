# Being told where to go, and seeing things instead of a picture (2026-08-15)

Status: **diagnostic**, on the already-consumed development seed. Nothing
admitted; `AgentBrain.bank` unchanged at `07319eb1`.

Two open items from the navigation record, which turn out to be one experiment.

**A goal could not be given.** The agent sought whatever had paid during
exploration. It could not be *asked* for anything, and one model served exactly
one goal — which is most of what a model is for.

**An observation could not have parts.** Every observation in this repository
has been atomic, so "what is in the scene" and "which scene is this" were the
same question.

Showing the goal *in the scene* answers both. The scene holds two markers —
where the agent is and where it has been asked to go — and is handed to two
agents differing in one respect: the **object** agent gets one event per
marker, the **scene** agent gets one event for the whole picture, which is what
every agent here has received until now. Same pixels, same frozen encoder.

## Result

Eight sampled worlds. Trained on goals 0–3; asked for goals 4–7 with **no
further exploration at all**.

| | object | told the goal | scene | wrong goal | random | best available |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| goals trained on | 0.667 | 0.696 | 0.618 | 0.141 | 0.134 | 0.733 |
| **goals never trained on** | **0.569** | 0.592 | **0.125** | 0.135 | 0.116 | 0.646 |

**A goal it was never trained on costs no new experience.** The object agent
reaches 88% of the best available return on held-out goals; the scene agent
reaches 19%, which is where acting blindly lands.

The scene agent is not broken — on goals it trained on it gets 84% of optimal.
It simply has nothing to carry across. Its model is a graph over pictures, and
a new goal is a region of that graph it has never been in.

**It is really using what it is shown.** Pointed at some other place instead of
the one in the scene, it drops to 0.135 — indistinguishable from acting
blindly (0.116). **And working out which marker it is costs little**: handed the answer
it gets 0.592 against 0.569, so identification is about 4% and the rest of the
gap to optimal is the model and the planner.

## Two things the scene had to be, and why

**Both markers share a colour.** Drawn in different colours, "goal at place
three" and "self at place three" encode to different events, the two objects
live in incomparable alphabets, and the agent can never tell that it has
arrived. One colour puts every marker in the same eight-place alphabet, and the
frontend separates them by **connected component** rather than by hue. Measured:
the slot alphabet is exactly **8**, and the slots read off any scene are exactly
the set of occupied places, whatever role each plays.

**Slot order is by position and means nothing.** It flips as the agent moves
past the goal, so the index cannot stand in for identity. Which object you
*are* is worked out from the only evidence available: across a wandering
episode one marker visits many places and the other stays, so the goal is the
place present in every observation. Identified in 24 of 24 exploration
episodes.

## Atomic reading is lossy, not merely large

The obvious story is that sixty-four scenes are simply more than eight places.
The measurement says something sharper. With identical markers, *agent at a,
goal at g* is **the same picture** as *agent at g, goal at a* — so sixty-four
configurations are **thirty-six** pictures, and a reactive reader of pictures
cannot tell the two apart even in principle.

Decomposition recovers what the single picture loses, because the objects are
individuated by **behaviour over time**: one of them moves when you act. That
is a better reason to have objects than counting states.

It also means the scene agent is handicapped by two things at once — a bigger
graph *and* an ambiguous observation — and the 0.125 should be read as the sum
of both rather than as compositionality alone.

## Two errors of mine, and how they surfaced

**The ceiling assumed the agent could stand still.** The sampler guarantees a
holding action at the task's *own* goal, not at every place that might be
asked for, so arrival-then-stay overstated what was achievable and the agent
read as failing when the ceiling was wrong. It is now exact finite-horizon
dynamic programming, which is why the optima here (0.73, 0.65) are lower than
the navigation record's.

**Goal identification spent a step outside the accounting** and gave up when a
single probe was inconclusive — which happens whenever the probing action
happens to hold still. It now narrows the intersection while acting, scoring
every step, and cycles actions so a self-loop cannot stall it.

## What is honestly weak

**The frontend does the segmentation, and does not learn it.** Connected
components on a colour mask is an encoder change, which `AGENTS.md` explicitly
allows — "simultaneous streams remain separately bindable rather than blindly
averaged" — but nothing here discovers that the scene *has* parts. It is told
where to cut.

**Two objects, one static.** Identity is worked out from persistence, which
works because the goal does not move. A second moving object, or a distractor,
breaks that rule outright and nothing here proposes a replacement.

**The object agent is at 88% of optimal, not at it.** Roughly 4% is
identification; the rest is model coverage and places with no holding action.

**Goals are places, not descriptions.** "Go where this marker is" is the whole
goal language. Nothing here handles a goal that is a property, a relation, or a
conjunction.

**The model is still per-task and discarded.** It is not compiled, not
admitted, and not composed — the accumulation machinery and this have still not
met.

**Eight worlds, one seed, no holdout.**

## Reproducing

```bash
PYTHONPATH=src .venv/bin/python -m experiments.brainworkshop_canonical.object_navigation
```

About ninety seconds.

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_object_navigation.py -q
```

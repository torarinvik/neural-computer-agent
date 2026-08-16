# Being asked for a relation nobody ever paid you for (2026-08-16)

Status: **diagnostic**, replicated at two seeds and two task counts. Nothing
admitted; `AgentBrain.bank` unchanged at `07319eb1`.

## A correction before the result

The successor transfer record left this as the open item:

> **Cumulants are hand-chosen.** `phi` is the place indicator [...] no
> property, no relation, no conjunction over anything but places.

That is wrong as written, and getting it right is what made this experiment
worth running. **A weight vector over places was never only "go to place p".**
Any *subset* of places is a vector, so disjunction and avoidance were already
expressible -- the same record measured both. Even "be in the same row as place
four" is a place-vector, because the satisfying set is fixed once the other
place is fixed.

The real ceiling is that **the satisfying set has to be constant**. It stops
being constant the moment the thing the goal refers to moves. "Stay next to
that marker" is not a set of places; it is a set of *configurations*, and it
changes under the agent's feet every time the marker does.

So the target here moves on its own -- a fixed circuit, deterministic,
uncontrollable -- and the state becomes the pair. The escalation costs a
cumulant matrix and nothing else: `successor_features`, generalised policy
improvement and the library all took it unmodified, because every one of them
was written against a cumulant matrix rather than against places.

## Result

Four worlds. Policies induced for `same`, `same_row`, `adjacent`; the agent is
then asked for `same_column`, `diagonal`, `opposite`, which were **never a
reward signal** -- only ever occupancies nobody asked about at the time. No
further experience. Normalised so the worst achievable is 0 and exact
finite-horizon optimal is 1.

| relation | held out | gpi | best stored | replan | place | random | constant-place rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| same | no | 0.933 | 0.933 | 0.933 | 0.206 | 0.184 | 0.125 |
| same_row | no | 0.964 | 0.964 | 0.964 | 0.423 | 0.348 | 0.375 |
| adjacent | no | 0.993 | 0.993 | 0.995 | 0.530 | 0.399 | 0.500 |
| **same_column** | **yes** | **0.988** | 0.840 | 0.998 | 0.404 | 0.407 | 0.375 |
| **diagonal** | **yes** | **0.677** | 0.319 | 0.992 | 0.281 | 0.191 | 0.250 |
| **opposite** | **yes** | **0.754** | 0.261 | 0.887 | 0.205 | 0.198 | 0.125 |
| | trained | 0.963 | 0.963 | 0.964 | 0.386 | 0.310 | |
| | **held out** | **0.806** | 0.473 | 0.959 | 0.296 | 0.265 | |

**Pairs are necessary, decisively.** The place representation gets 0.296-0.386
against 0.806-0.963 for pairs, and sits barely above acting blindly
(0.265-0.310). It is given the fairest weight vector available to it -- for
each place, how often the relation would hold if the marker were anywhere --
and it still cannot follow something that moves. This is the control that
decides whether the escalation was needed, and it says yes.

**A relation never rewarded transfers.** 0.806 against 0.473 for the best
single stored policy, a factor of 1.7.

**And it costs more than a held-out goal did.** On trained relations stitching
matches re-solving exactly (0.963 against 0.964). On held-out relations it does
not: 0.806 against 0.959. That gap is the finding.

## The PGM warning, reproduced

Barrett et al. (2018) is the reason the split was made this way: networks that
interpolate happily collapse when a held-out *attribute* appears at test, and
every earlier split in this repository was interpolation -- new goals drawn
from the same family as the trained ones, where stitching matched re-solving.

Change the split to held-out *relations* and the same machinery loses 0.16 of
optimal. `diagonal` is the worst at 0.677 against a re-solved 0.992: the stored
policies were built for `same`, `same_row` and `adjacent`, and none of them
spends much time in diagonal configurations, so there is little occupancy for a
diagonal weight vector to find. Transfer is bounded by what the stored policies
happened to visit, and a relation orthogonal to all of them is where that bites.

## Replication

| seed | worlds | trained gpi | trained replan | held gpi | held single | held replan | held place |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 41 | 4 | 0.963 | 0.964 | 0.806 | 0.473 | 0.959 | 0.296 |
| 41 | 6 | 0.974 | 0.975 | 0.849 | 0.482 | 0.913 | 0.312 |
| 137 | 4 | 0.957 | 0.958 | 0.813 | 0.471 | 0.918 | 0.316 |
| 137 | 6 | 0.927 | 0.927 | 0.769 | 0.447 | 0.910 | 0.327 |

Every ordering above holds in all four. The held-out gap ranges 0.06-0.15 and
is always positive.

## Three relations were thrown out before any of this

`above`, `left_of` and `far` are each satisfied **0.625** of the time by
standing in one fixed place forever. With them in the set, the place control
scored **1.000** on `above` -- the same as the relational agent -- because the
task never required looking at the marker at all. That is the constant-answer
trap the rule sampler already guards against, arriving in a new costume.

`constant_place_rate` is now the guard, `CONSTANT_PLACE_LIMIT` is 0.5, and a
test asserts every relation in use clears it. The surviving set rates 0.125 to
0.500.

## What is honestly weak

**Identification is handed over** by the declared oracle. Two markers that both
move is precisely the correspondence case the identity record measures at
0.617-0.625, and letting it vary here would move two axes at once. The target's
*place* still comes from pixels; only which slot is which is given. An
end-to-end agent would compound the two.

**The relations are hand-written**, exactly as the place cumulants were. This
widens the goal language by one level of structure; it does not discover the
level. Nothing here proposes where a relation vocabulary would come from.

**Six relations, one geometry, two objects.** No conjunctions of relations, no
three-object relations, no relations over properties other than position.

**The target's motion is deterministic and uncontrollable.** That keeps the
ceiling exact backward induction. A target that reacts to the agent is a game,
not a task, and is untouched.

**The configuration space is 64 states and coverage is 0.978.** The transfer
result is about representation, not about a model with holes in it -- but it
also means nothing here is tested under partial knowledge.

**Four to six worlds, two seeds, no holdout block.**

## Reproducing

```bash
PYTHONPATH=src .venv/bin/python -m experiments.brainworkshop_canonical.relational_transfer --tasks 4
```

About two and a half minutes.

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_relational_transfer.py -q
```

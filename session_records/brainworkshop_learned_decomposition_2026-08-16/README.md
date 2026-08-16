# Finding where to cut, instead of being told (2026-08-16)

Status: **diagnostic**, on the already-consumed development seed. Nothing
admitted; `AgentBrain.bank` unchanged at `07319eb1`.

The caveat this closes, quoted from the object-navigation record:

> **The frontend does the segmentation, and does not learn it.** Connected
> components on a colour mask is an encoder change, which `AGENTS.md`
> explicitly allows [...] but nothing here discovers that the scene *has*
> parts. It is told where to cut.

MONet, IODINE and Slot Attention discover parts by competing to **reconstruct**
the image. All three are gradient-trained autoencoders with a decoder, and this
repository has neither. What it has that they mostly do without is **actions
and time**, so the criterion changes: a cut is good when the pieces move
independently, and that is measured as the cost of writing the dynamics down.

Description bits plus error bits, per part, in the same sense used everywhere
else here. Connected components is one entrant among seven and is not
privileged in the scoring.

## Result

Eight sampled worlds, four wandering episodes each, every candidate scored on
**the same** frames and actions.

| cut | parts | alphabet | description | error | total |
| --- | ---: | ---: | ---: | ---: | ---: |
| **components** | 2.0 | 8.0 | 134.6 | **0.0** | **134.6** |
| whole | 1.0 | 23.2 | 315.2 | 44.5 | 359.8 |
| halves_horizontal | 2.0 | 24.8 | 444.8 | 244.9 | 689.7 |
| cells | 9.0 | 9.0 | 334.0 | 374.8 | 708.9 |
| halves_vertical | 2.0 | 24.5 | 436.5 | 278.0 | 714.5 |
| scatter | 2.0 | 46.5 | 769.2 | 108.6 | 877.8 |
| quadrants | 4.0 | 21.5 | 552.0 | 551.1 | 1103.1 |

**Chosen in 8 of 8 worlds, at 2.7x cheaper than reading the scene whole, and
with exactly zero error bits.** One part per marker is not assumed here; it
wins.

The two failure directions are punished by different terms, which is what makes
the criterion more than a preference:

- **too coarse** -- `whole` pays 315 description bits for an alphabet of 23,
  *and* 44.5 error bits, because with identical markers the scene does not
  determine what happens next. The aliasing the previous record noted in
  passing shows up here as a price.
- **too fine** -- `cells` has the second-cheapest tables in the whole family
  (334 bits for nine binary parts) and the second-worst error, because whether
  a cell becomes occupied is a fact about a *different* cell.
- **right count, no structure** -- `scatter` deals pixels into two groups at
  random. Same part count as components, 6.5x the cost.

## It refuses to decompose when there is nothing to decompose

Hold the goal still across every episode and one marker never changes. Then:

| cut | total bits |
| --- | ---: |
| whole | **78.0** |
| components | 90.0 |

Reading the scene whole is correctly the cheaper answer, because the second
part costs bits and earns none. This is the same fact the object-navigation
record measured from the other side, where the scene agent did fine on goals it
had trained on. The criterion is not a bias toward more parts.

## An error worth recording

The component cut initially carried 83 error bits, which should have been
impossible -- the goal never moves and the agent's transitions are
deterministic. Tracking was not the cause: the goal track never once switched
index over 16 episodes.

The cause was that slot order is positional and therefore means nothing *across
episodes*, so part 0 was the agent in one episode and the goal in the next, and
the pooled tables mixed two objects under one key. Canonicalising part order
within each episode by how often the part changes symbol -- the same kind of
relabelling the Mealy machines get -- took the error term to **zero**. It is
applied identically to every candidate, so it cannot favour one.

## What is honestly weak

**The search is over a fixed family, not over all partitions.** Seven
candidates, hand-written. This *selects* a decomposition against real
alternatives; it does not invent one, and a scene whose right cut is not in the
family would be missed silently.

**Components needs alignment to be scoreable at all**, because its parts arrive
in positional order. `slot_alignment` is load-bearing here, and a cut that
could not be tracked would lose for a reason that is not about the cut.

**One frontend, one scene type.** Two or three identical square markers on a
3x3 grid. Nothing here says the criterion survives occlusion, deformation,
lighting, or parts that are not spatially connected.

**Eight worlds, one seed, no holdout.**

## Reproducing

```bash
PYTHONPATH=src .venv/bin/python -m experiments.brainworkshop_canonical.learned_decomposition --tasks 8
```

About seven seconds.

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_learned_decomposition.py -q
```

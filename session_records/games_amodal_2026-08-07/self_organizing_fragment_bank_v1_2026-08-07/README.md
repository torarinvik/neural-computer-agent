# Promoted: self-organizing fragment bank (2026-08-07)

A fixed-size amodal plant holds two contradictory policies at once, with
the choice between them carried entirely by opaque fragments the system
fetches from its own bank — content learned from scalar outcomes,
addressing learned by the selector, nothing privileged at runtime.

Task: `choice` twins. Both variants render identically (one type-A and
one type-B item adjacent to the avatar); in A taking type-A pays +1 and
type-B costs -1, in B the roles swap. Navigation costs one step, so the
only learnable content is which type to take, and it is unavailable from
the screen.

Mechanism (the recipe forced by design laws F1-F13):
* fragment tokens initialised at event scale (F8);
* both contexts present from the first update, no blind warm-up (F8);
* laggard-preferential context sampling (F10);
* selector diversity penalty against collapse (probe 11);
* distinct peaked selection-logit initialisation (F12);
* **scaffolded addressing** — 1200 updates of oracle-driven selection
  during which the learned selector imitates the assignment by KL, then
  1200 updates with the selector in full control (F13);
* ignorance objectives (withheld/decoy) throughout.

Command (per seed):

```bash
uv run python -m experiments.games_amodal.fragment_bank \
  --seed <seed> --suite twins --oracle-updates 1200 --updates 2400 \
  --balance-contexts --selection-diversity 2.0 --selection-init-scale 3.0 \
  --warm-updates 0 --batch-size 32 --steps 48 \
  --ignorance-weight 0.5 --ignorance-every 3
```

## Result (both seeds)

| condition (mastery) | 69316 A/B | 69317 A/B |
| --- | ---: | ---: |
| own fragments, learned selection | **1.000 / 1.000** | **1.000 / 1.000** |
| bank withheld | 0.188 / 0.445 | 0.383 / 0.125 |
| noise decoy (matched norm) | 0.219 / 0.219 | 0.203 / 0.211 |
| **cross-fed (other context's fragments)** | **0.000 / 0.000** | **0.000 / 0.000** |
| learned selection (post-handover) | [1,2] / [4,5] | [0,2] / [4,5] |
| replayed examples | 0 | 0 |

Cross-feeding at exactly 0.000 in both directions on both seeds is the
decisive signature: the fragment does not merely enable competence, it
specifies *which* competence runs — the agent systematically takes the
wrong item when handed the wrong program. Withheld and decoy conditions
collapse to chance or below. The selector, in sole control for the final
half of training, maintains disjoint assignments.

## Rejected alternatives (preserved here)

* `rejected_unstable_selection.json` — learned selection with zero-init
  logits: diversity penalty produced disjoint assignments but the
  unstable early mapping left the plant fragment-blind (all four audit
  conditions identical).
* `rejected_stable_init_only.json` — distinct peaked init without
  handover: blindness broken (conditions differ) but winner-take-all
  persisted (0.242 / 0.781).

## Claim boundary

Promoted: two contradictory contexts, one plant, opaque context-carried
program selection, learned content and learned addressing, zero replay,
two seeds. Not promoted: more than two contexts; fragment *sharing* or
compositional reuse (this rung has no shared structure to find);
compounding across a growing library; content-addressed retrieval from
observations rather than per-context selection logits; and any claim that
the oracle scaffold can be removed entirely rather than scheduled.

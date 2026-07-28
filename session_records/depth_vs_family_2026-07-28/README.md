# Transfer is a first-composition effect

The project's central hope was that a retained library makes each later skill
cheaper, and that the discount grows as the library grows. Measured under
exact-zero gating across four ancestry depths, the first half is true once and
the second half is false: **going from one skill to two makes the next skill
dramatically cheaper, and going from two to three, or three to four, does
nothing measurable.**

Everything here is measured on rectified slot gates. That matters: the previous
measurements were taken while every rung was quietly damaging its nearest
neighbour, so an apparent transfer advantage could not be separated from a slot
borrowing structure it was also degrading. With the gate able to shut exactly,
degradation is at zero and the advantage is whatever it really is.

## Result

Paired within seed; the deep arm has exactly one more rung of ancestry than the
shallow arm, both replay the same skills for the same experience.

| new task | family | ancestry | pooled paired delta | sign test | seeds |
|---|---|---|---:|---|---:|
| `visible_context_xor` | visible | **1 → 2** | **+0.0870** | 105W/39L, p=3.5e-8 | 16 |
| `context_rule_xor` | hidden rule | **1 → 2** | **+0.0648** | 76W/20L, p=7.3e-9 | 12 |
| `contextual_composition` | hidden rule | 2 → 3 | −0.0163 | 22W/33L, p=0.18 | 8 |
| `context_identity_and` | hidden rule | 3 → 4 | −0.0002 | 37W/45L, p=0.44 | 12 |

Restricting to the unsaturated part of each curve, where both arms are still
learning, does not change it: +0.0975, +0.0656, −0.0173, +0.0189.

## Why the two families matter

Ancestry depth and task family were confounded in the earlier work: the rung
that transferred was a fully visible zero-shot composition, and the rungs that
did not were hidden-rule few-shot ones. The second row above breaks that. A
**hidden-rule** task at ancestry 1 → 2 transfers at +0.0648 with p = 7.3e-9,
while the **same family** at 2 → 3 and 3 → 4 gives −0.0163 and −0.0002.

The effect is set by depth, not by family.

Which family is extensible is itself forced. The gate suite requires reversing
the audited variable to flip at least 80% of actions. In the visible family
`correct = f(identity, context)` that admits only XOR-type `f`, and those are
exhausted — `and` and `or` flip just 50%. In the hidden-rule family
`correct = rule ^ f(identity, context)` reversing the rule flips every action
for any `f`, so that family extends indefinitely. Both new primitives here,
`context_identity_and` and `context_identity_or`, come from it.

## Two estimators, and which to trust

At ancestry 1 → 2 on the hidden-rule task the pooled accuracy advantage is large
and highly significant (+0.0648, p = 7.3e-9) while the interpolated
experience-to-threshold ratio is 1.059 and not significant. They disagree
because the deep arm's advantage is concentrated mid-curve — +0.119 and +0.141
at 72 and 96 updates — and the two curves converge again by the time both cross
0.85. A threshold ratio samples one point of a curve; the pooled paired delta
uses all of it, and here it has far more power.

The threshold ratios, for completeness: 1.401 (p = 0.013) at 1 → 2 visible,
1.059 at 1 → 2 hidden, 0.925 at 2 → 3, 1.000 at 3 → 4. Same ordering, less
resolution.

## What this retires

The earlier reading was that the transfer advantage shrank across rungs
(1.231 → 1.083) and that interference was the binding constraint. Removing
interference did not restore compounding. It sharpened the finding: there was
never a compounding curve to recover, only a first-composition effect that had
been measured twice at different depths.

The one genuinely new thing a second skill buys appears to be the *ability to
compose at all* — the jump from a single mapping to a mapping plus a context is
what makes a composed task learnable cheaply. Adding further primitives to the
library does not repeat that.

## Method notes

- 0 dead slots in the 112-run rung-4 race and 0 in the 576-run rung-3 sweep,
  confirming the gate warmup at a scale the earlier 4.2% figure was measured at.
  Two dead slots appear in the smallest-budget cells of the 1 → 2 hidden sweep,
  where 5% of 24 updates rounds to a single warmup update.
- Both the original 1 → 2 hidden grid and the 3 → 4 grid saturate above 512
  updates; the informative region is the low-budget part, reported separately
  above.
- Each depth necessarily uses a different new task, since a skill cannot be
  learned twice. The family control addresses the largest version of that
  confound but not all of it.

## Cue capacity is the next hard limit

Placing the two new primitives required a preflight separability search rather
than judgement. Candidate slots in the bottom band scored 0.22 to 0.58 against
the two cues already there — below the threshold that predicts interference —
while top-band candidates reached 1.01 to 1.09 regardless of area or intensity.
Position band dominates; area and intensity are weak secondary axes.

The frozen encoder can only tell so many operations apart, and that budget is
close to spent. Its global average pool is the reason. Any further extension of
this ladder needs either a cue code the pooled encoder can resolve, or an
encoder that keeps some spatial detail.

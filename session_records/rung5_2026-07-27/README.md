# A rung that costs the earlier skills nothing

Earlier today's record established that this ladder degrades: each new
primitive took roughly 1.8 accuracy points off `visible_context`, its nearest
neighbour, against a 0.90 gate. That trajectory predicted the ladder would stop
working in about three more rungs. The conclusion was that interference, not
learning speed, was the binding constraint.

That constraint is now removed. Changing one thing — making a successor slot's
gate able to shut *exactly* — takes the per-rung cost from 1.4 accuracy points
to zero, at no cost in new-skill accuracy.

## The ladder, on identical evaluation lifetimes

| controller | binary_mapping | visible_context | visible_context_xor | composition |
|---|---:|---:|---:|---:|
| rung 3 (inherited) | 0.9781 | 0.9359 | 0.9250 | 0.4797 |
| rung 4, sigmoid gate | 0.9736 | 0.9224 | 0.9235 | 0.9949 |
| **rung 4, rectified gate** | **0.9781** | **0.9362** | **0.9250** | **1.0000** |

The rectified rung adds a fourth primitive at 100% while leaving all three
inherited skills unchanged — two of them to four decimal places. The sigmoid
rung, promoted earlier with every gate passing, quietly costs 1.35 points of
`visible_context`, 0.45 of `binary_mapping`, and 0.15 of the XOR.

## Why exact zero is the whole mechanism

A slot adds `adapter(features) * gate(features)` to the intention on **every**
event, including events belonging to skills it has nothing to do with. A
sigmoid gate is bounded away from zero, so that perturbation is never absent —
only small. Small and always-on is exactly the shape of a quantity that
accumulates.

A rectified gate can output exact zero, and then the slot is not approximately
absent but genuinely absent: the inherited controller is frozen and
bit-identical, so an exactly-zero residual leaves old behavior untouched by
construction rather than by training pressure.

Measured selectivity on the promoted controller: the slot is open on **100%**
of its own task's events and exactly shut on **96.6–98.7%** of each old skill's.
A sigmoid slot is exactly shut on 0.0%, structurally.

## Three things that did not work, and why they are informative

**Pricing the gate opening.** A penalty on the mean opening drove it from 0.410
to 0.0101, a fortyfold cut, and retention got *worse* (−0.0132 → −0.0327). The
adapter simply grew its output to compensate: the residual norm fell only 0.80
to 0.48 while the norm on the new task rose 4.76 to 6.06. Pricing a proxy that
the model can route around achieves nothing. The residual norm is the quantity
that matters, and even pricing that correctly only trades new-skill accuracy
away.

**Raising the retention price.** A fixed weight does reduce degradation, but the
frontier is hard: weight 2.0 halves it at no accuracy cost, weight 8 nearly
removes it while dropping the new skill to 0.63, and weight 32 removes it with
the new skill at chance. There is no setting that gets both.

**Set-point control on the price.** Holding each old skill at the level it was
inherited at, with a proportional gain on its shortfall, is flat-to-worse in
gain (−0.0063 at gain 0, −0.0096 at gain 160). More pressure on the same
mechanism cannot buy selectivity that the mechanism cannot express.

All three failures share a cause: they try to *pay* for locality that the
architecture cannot represent. The rectified gate makes locality representable,
and then it costs nothing.

## Holding across a further rung

Rung 5 trains `contextual_override` on the rectified four-skill controller,
replaying all four earlier primitives. Six seeds per gate mode.

| inherited skill | rectified | sigmoid |
|---|---:|---:|
| `binary_mapping` | −0.00005 (3/4 exactly zero) | −0.01439 (6/6 negative, p=0.031) |
| `visible_context` | −0.00004 (3/4 exactly zero) | **−0.02900** (6/6 negative, p=0.031) |
| `visible_context_xor` | +0.00016 | −0.00429 |
| `contextual_composition` | +0.00000 (4/4 exactly zero) | −0.00055 |
| new skill | 0.9999 | 0.9989 |
| exactly shut on old events | 99.1% | 0.0% |

The sigmoid arm loses 2.9 points of `visible_context` in this single rung —
accumulation accelerating as the trajectory predicted, and enough on its own to
push that skill through its gate. The rectified arm loses 0.004 points, **725×
less**, with a *higher* new-skill accuracy.

## What this rung is and is not

Rung 5 is an interference probe, not a promoted rung. `contextual_override`
fails `counterfactual_flip_at_least_80` and `reversed_few_shot_at_least_85` for
both gate modes at 100% normal accuracy, because its context-one branch is a
constant the learner can memorise instead of tracking the rule. That is a
property of the task, not of the method, but it means nothing here is promoted
and the retention numbers are what the rung is for.

Two further caveats. The rung-5 harness replays and evaluates every old skill at
one support outcome, while `contextual_composition` was acquired at two, so that
skill's retention gate is not required of either arm; its delta is still
measured like-for-like between parent and student. And rectified gates die on
about 8% of seeds (1 of 12 in a fresh panel, 2 of 6 at rung 5) — the classic
dying-rectifier failure, in which the gate shuts everywhere before it has
learned where to open, and is then permanently without gradient. It is trivially
detectable, since the slot is shut on 100% of its own task's events, but it is
an unfixed robustness cost.

## Robustness of the mechanism itself

A fresh panel of twelve seeds, rectified gate, rung 4:

- 11 live, 1 dead-gate (8%)
- **11/11 live seeds pass every gate**
- mean new skill 0.9929
- mean worst degradation across three skills −0.00095, exactly zero on 5/11
- mean exact-shut on old events 93.8%

## Promoted controller

`artifacts/checkpoints/unified_four_skill_rectified_seed8414.pt`, seed 8414,
6144 updates, rectified linear gate.

- new composition 0.9995 on 2,048 held-out lifetimes, all ten sub-gates
- deltas against inherited: `binary_mapping` 0.0, `visible_context` +0.0007,
  `visible_context_xor` −0.0001
- slot open on 100% of composition events, exactly shut on 96.6–98.7% of others
- removing only the operation cue drops the new skill to 0.5034
- inherited weights bit-identical; reloads on another machine and passes all
  four skills, composition at 1.0000

## Consequence

The extinction estimate is gone. At 1.8 points per rung `visible_context` was
about three rungs from failing its gate. At the rectified rung's measured cost
the quantity is not decaying at all, so the ladder's length is no longer bounded
by interference. What bounds it now is unknown, which is the right next question.

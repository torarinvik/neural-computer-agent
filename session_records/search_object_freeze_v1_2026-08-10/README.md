# The transition model was fine; the search wasn't (F109)

## Per-slot diagnosis

| slot | accuracy |
| --- | ---: |
| avatar row / col | 1.0000 / 1.0000 |
| nearest positive object row / col | 0.6722 / 0.6722 |
| nearest negative object row / col | 0.7728 / 0.7712 |

Avatar dynamics learned perfectly. The 0.5842 exact-state figure is entirely the
object slots, which are NOT learnable — "nearest object" changes discontinuously
when the avatar moves or an item respawns.

## The search defect it revealed

Beam search rolled all six slots forward, compounding 0.72/step into ~0.27 over
a depth-4 plan — planning against invented object positions. Items sit still
between respawns, so the observed layout is the better estimate.

| arm | held-out | twin | entry effect | % headroom |
| --- | ---: | ---: | ---: | ---: |
| ignorance 0.5 | -0.0217 | -0.0716 | +0.0499 | 22.0% |
| + freeze objects | -0.0205 | -0.0782 | +0.0577 | 25.4% |

Both seeds improve. Real, and small.

## Refuted hypothesis

Predicted: the "nearest object only" abstraction is the limit — complete with 1
item pair, incomplete with 3.

| item pairs | entry effect | % headroom |
| ---: | ---: | ---: |
| 1 (state complete) | +0.0450 | 19.8% |
| 2 | +0.0440 | 19.3% |
| 3 (state most incomplete) | +0.0762 | 33.6% |

Backwards. Outcome accuracy flat across counts (0.4320 vs 0.4291). Likeliest
reading — more items means more chances to eat, i.e. test-time reward density
rather than state completeness — is a hypothesis, recorded as one.

## Remaining ~75%: undiagnosed

Three candidates tested, none explains it: transition model (perfect where it
matters), search object rollout (3.4 points), state abstraction (refuted).
Outcome model at 0.4474 balanced is the obvious suspect but is not shown to be
binding. Two wrong diagnoses already recorded on this question (F101, F108);
a fourth guess is worse than saying so.

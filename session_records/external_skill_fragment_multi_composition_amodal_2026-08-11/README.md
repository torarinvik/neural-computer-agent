# Four-fragment closure audit — 2026-08-11

This is the next bounded step after the two-fragment trace-combiner result.
Four opaque fragments are acquired sequentially from fresh rendered outcomes.
Each primitive must pass stable held-out mastery before its coefficient row and
the learned shared-basis prefix are protected. A separate external trace
combiner and decoder then learn one held-out four-fragment program. The parent
controller remains frozen throughout.

| seed | primitive stable bits | inherited composition | fresh composition | fresh/inherited | wrong order | zero codes | missing evidence | shuffled |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 69316 | 12,288 / 12,288 / 6,144 / 12,288 | 0.9531 | 0.9870 | 2.00x | 0.5677 | 0.6250 | 0.4714 | 0.3776 |
| 69317 | 12,288 / 12,288 / 24,576 / 6,144 | 0.9583 | 0.8620 | 2.00x | 0.4844 | 0.6536 | 0.5000 | 0.4167 |

Every promotion gate passed on both seeds: all four primitives mastered and
were retained, composition was stable, transfer was positive, order sensitivity
was present, no-fragment and missing-evidence controls failed as expected,
shuffled outcomes were rejected, the parent digest stayed unchanged, the bank
survived exact persistence and corruption rejection, routing resolved to the
opaque target order, and no examples were replayed.

The rejected diagnostic rung is part of the result: using one decoder objective
for both a primitive and a longer composition entangled the primitive and
failed seed-69317 retention. Separating atomic acquisition from composition
acquisition fixed it at the next exposure rung. This is the stronger foundation
for continual learning even though the result remains bounded and does not
establish arbitrary program induction or general continual learning.

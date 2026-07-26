# Interleaved curriculum screen

Date: 2026-07-26

## Question

Could a schedule that alternates old utility, reliability transfer, and old
utility return make a short prefix informative for both transfer and retention,
without changing the physical task weights or verifier budget?

## Harness result

`interleaved_reliability` repeats this three-round cycle:

1. `old_equal` — `(0.5, 0.5, 0.0)`;
2. `reliability_dominant` — `(0.3, 0.3, 0.4)`;
3. `old_return` — `(0.5, 0.5, 0.0)`.

At 12 rounds, every phase was visited four times. A 12-round prefix matched
the first 12 trace rows of the corresponding 15-round run bit-for-bit, while
remaining explicitly non-graduating. The 12-round smoke run passed physical
persistence, parity, and inherited-primitive retention gates.

## Promoted comparison

The previously strong all-7073 value-diverse trajectory was replayed for 54
physical rounds with identical controller, random streams, task weights,
strategy capacity, and verifier cost. Only temporal ordering changed.

| Schedule | Reward-informative soft pairs | Bits / informative pair | Reliability target | Old-return target |
|---|---:|---:|---:|---:|
| Blocked standard | 56.6% | 214.4 | 41.7% | 95.8% |
| Interleaved | 5.7% | 2,144.0 | 9.7% | 9.7% |

The interleaved arm passed generic safety/accounting gates, but it is about
ten times less information-efficient and loses the retained-return advantage.
It is therefore rejected as a learning schedule. No reward-shuffle control is
warranted because there is no positive capability result to validate.

## Consequence

Keep exact-prefix reporting; it prevents false successive-halving claims. Do
not use simple interleaving to shorten selection races. The current frontier is
to expose switching and retention earlier without breaking the contiguous
trajectory that appears necessary for strategy-memory ignition.

# The gate at N=256 (F87)

256 novel families acquired sequentially through one frozen plant, 2 seeds.

    amortised_bank.py --pool 4096 --train-updates 40000 --sequential 256 --retrieval

## All three clauses hold at 4x the previous bank size

- (a) 256/256 mastered, both seeds
- (b) retention drift max 0.0 across all 256 entries
- (c) no acquisition drift: first-64 vs last-64 is 2.7 -> 3.5 and 7.0 -> 4.7
  (one rises, one falls) against cold ~51. Still 10-15x cheaper.

## Retrieval

| N | key | key+verify | linear scan |
| ---: | ---: | ---: | ---: |
| 64 | 1.000 | 1.000 | 0.969 |
| 128 | 0.996 | 1.000 | 0.953 |
| 256 | 0.988 | 0.994 | 0.918 |

Retrieve-then-verify is best at every size, holds 0.994 at N=256 at a CONSTANT
4 plant passes, while the 256-pass linear scan has decayed to 0.918.

## The extrapolation would have been wrong

| N | key gap | consequence gap | stranger key similarity |
| ---: | ---: | ---: | ---: |
| 8 | 0.325 | 0.571 | 0.667 |
| 64 | 0.128 | 0.356 | 0.862 |
| 256 | 0.068 | 0.258 | 0.923 |

Key-gap decrements per doubling: -0.109, -0.049, -0.039, -0.033, -0.027 —
decelerating, roughly halving. The linear-in-log projection (zero in the low
thousands) is wrong; the curve approaches a small positive asymptote. Declining
to extrapolate from four points was correct.

The shrinking gap does NOT break retrieval (ranking survives to 0.994). It DOES
break threshold-based reuse-or-mint on keys alone: a never-seen family matches
its nearest key at 0.923. Consequence verification still separates (0.258), so
the verify step is load-bearing, not cautious.

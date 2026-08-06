# Repeated nonstationary growth: length 6 → 8 → 10 (2026-08-06)

This is the first sequential multi-shift audit. Two capabilities are learned
on length-six episodes. Eight new capabilities are then acquired from fresh
length-eight episodes, followed by ten more from fresh length-ten episodes.
The controller, earlier route state, and earlier credit heads remain frozen
through both shifts; no earlier examples are replayed.

## Promoted result

| metric | seed 69316 | seed 69317 |
| --- | ---: | ---: |
| phase-1 minimum route selection (length 8) | 0.9219 | 0.8906 |
| phase-2 minimum route selection (length 10) | 0.8906 | 0.9063 |
| old route/permutation | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| phase credit / combined credit | 1/1, 1/1 / 1.000 | 1/0.875, 1/1 / 0.950 |
| full-bank protection/reversal/recovery | passed | passed |
| reward-shuffled false selections | 0 | 0 |
| replayed examples | 0 | 0 |

Both seeds pass phase-wise route recovery, causal extensions, prior-extension
ordering, all-shift credit, permutation, full-bank protection, isolated
reversal/recovery, and the zero-centered antithetic null.

## Claim boundary

This promotes two sequential temporal distribution shifts over a fixed
20-capability bank. It is the strongest current bounded continual-learning
result, but it does not establish unbounded expansion, arbitrary program
induction, or general continual learning.

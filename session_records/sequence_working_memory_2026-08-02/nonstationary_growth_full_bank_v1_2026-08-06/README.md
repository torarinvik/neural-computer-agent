# Nonstationary full-bank growth (2026-08-06)

This combines the two strongest bounded tests. Two protected capabilities are
learned on length-six episodes; 18 new capabilities are then acquired from
fresh length-seven episodes, filling the 20-family external bank. The
controller, old routes, and old credit state remain frozen. No length-six
examples are replayed after the shift.

## Promoted result

| metric | seed 69316 | seed 69317 |
| --- | ---: | ---: |
| total capabilities | 20 | 20 |
| minimum new-route selection | 0.9688 | 0.9219 |
| old-route accuracy | 1.0000 | 1.0000 |
| candidate permutation | 1.0000 | 1.0000 |
| old/new/combined credit | 1/1/1 | 1/0.9444/0.9500 |
| replayed examples | 0 | 0 |
| reward-shuffled false selections | 0 | 0 |

Both seeds pass new-route causality, prior-extension ordering, full-bank
protection, single-target reversal, fresh recovery, and the antithetic null
control. This is the strongest current bounded continual-learning result:
capacity pressure and one temporal distribution shift coexist without
catastrophic forgetting in the external state.

## Claim boundary

This is still one controlled temporal shift over a closed 20-family bank. It
does not establish repeated arbitrary distribution shifts, unbounded dynamic
expansion, arbitrary program induction, or general continual learning.

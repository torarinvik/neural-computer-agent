# Nonstationary full-bank length-six to length-eight growth (2026-08-06)

This magnitude-of-shift rung freezes two capabilities learned on length-six
episodes, then acquires 18 fresh capabilities from length-eight episodes,
filling the 20-family external bank. The controller, old routes, and old
credit state remain frozen. No old examples are replayed after the shift.

## Promoted result

| metric | seed 69316 | seed 69317 |
| --- | ---: | ---: |
| total capabilities | 20 | 20 |
| minimum new-route selection | 0.8906 | 0.8281 |
| old-route accuracy | 1.0000 | 1.0000 |
| candidate permutation | 1.0000 | 1.0000 |
| shifted credit old/new/combined | 1/1/1 | 1/1/1 |
| replayed examples | 0 | 0 |
| reward-shuffled false selections | 0 | 0 |

Both seeds pass new-route causality, prior-extension ordering, full-bank
protection, isolated reversal/recovery, and the antithetic null control. The
larger temporal shift lowers the weakest route but remains above the promotion
gate, showing that this external state is not limited to adjacent episode
lengths.

## Claim boundary

This is still one controlled shift over a closed 20-family bank. It does not
establish repeated arbitrary shifts, unbounded dynamic expansion, arbitrary
program induction, or general continual learning.

# Retrieval by consequence: clause (c) becomes falsifiable (F85)

F83 could not fail clause (c) because entries were handed to the correct family
by construction. Retrieval closes that: score every stored entry by how well it
predicts held-out transitions of the task at hand, take the best (F44's
consequence probing).

    amortised_bank.py --pool 4096 --train-updates 40000 --sequential 64 --retrieval

| bank N | accuracy | chance | margin | in-bank | outside-bank | forward passes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1.000 | 0.125 | 0.564 | 1.000 | 0.429 | 8 |
| 16 | 1.000 | 0.062 | 0.456 | 1.000 | 0.496 | 16 |
| 32 | 1.000 | 0.031 | 0.400 | 1.000 | 0.596 | 32 |
| 64 | 0.969 | 0.016 | 0.365 | 1.000 | 0.642 | 64 |

Both seeds 0.9688 at N=64.

## The discrimination null

A task NOT in the bank must score low against every entry, or "retrieval
accuracy" is satisfied by a system that always returns something. In-bank 1.000
throughout; strangers rise 0.429 -> 0.642. The gap shrinks monotonically:
0.571, 0.504, 0.404, 0.358.

## Two limits, kept separate

**Measured**: retrieval is O(N). 64 forward passes to identify a known task vs
2.7-7.0 update steps to mint a fresh entry — at N=64 recognising a task already
costs more than learning it. Naive linear banks do not scale; content-addressed
keys are the fix and have never been wired in (open weakness 8).

**Projected, not measured**: discrimination degrades ~0.07 per doubling; a
linear-in-log extrapolation reaches zero in the low thousands. Four points, two
seeds — the direction is the claim, not the intercept. The runner-up margin's
decrements are decelerating (-0.108, -0.056, -0.035), which would push any
crossing further out.

## Gate status

(a) 64/64 mastered both seeds. (b) retention drift exactly 0.0. (c) passes on
accuracy, fails on cost efficiency — and can now fail at all, which F83's
version could not.

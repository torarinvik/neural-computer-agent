# Nonstationary length-six to length-seven growth (2026-08-06)

This is the promoted temporal distribution-shift audit. Two capabilities are
learned on length-six episodes. The encoder, old routes, and old credit state
are then frozen while eight new external routes and isolated credit heads are
acquired from fresh length-seven episodes. No length-six examples are replayed
after the shift.

The shuffled extension control uses an antithetic null: each query is paired
with exactly contradictory scalar outcomes, so the expected credit is exactly
zero rather than merely zero in expectation over a small random batch. This is
trainer-side control logic and never enters the deployed controller.

## Promoted result

| metric | seed 69316 | seed 69317 |
| --- | ---: | ---: |
| old route accuracy | 1.0000 | 1.0000 |
| candidate permutation | 1.0000 | 1.0000 |
| minimum new-route selection | 0.9688 | 0.9375 |
| old/new/combined credit accuracy | 1/1/1 | 1/1/1 |
| reward-shuffled extension selection | 0.0000 for all | 0.0000 for all |
| replayed examples | 0 | 0 |

Both seeds pass causal new-route ablations, prior-extension ordering,
retention-safe reversal and recovery, and zero replay. The prior random-shuffle
version is retained as a regression control in
`nonstationary_growth_episode6_to7_rejected_v1_2026-08-06/`.

## Claim boundary

This promotes one controlled temporal distribution shift with isolated
external state. It does not establish arbitrary distribution shifts,
unbounded memory growth, arbitrary program induction, or general continual
learning.

# Reusable external growth prior: length 6 → 8 → 10 → 12 (2026-08-06)

This audit adds a copy-on-write growth prior to the promoted three-shift
schedule. New route adapters in later shifts reuse the averaged representation
of earlier external adapters while their capability-specific score head is
reset to a neutral state; each acquired adapter then remains an isolated
mutable artifact. The shared controller, old route state, and old credit heads
remain frozen, and no earlier examples are replayed.

## Promoted result

| metric | seed 69316 | seed 69317 |
| --- | ---: | ---: |
| phase-1 minimum route selection | 0.9219 | 0.8906 |
| phase-2 minimum route selection | 0.8750 | 0.8906 |
| phase-3 minimum route selection | 0.9375 | 0.8438 |
| old route/permutation | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| credit old / shift 1 / shift 2 / shift 3 / combined | 1 / 1 / 1 / 1 / 1.000 | 1 / 0.875 / 1 / 1 / 0.969 |
| full-bank protection/reversal/recovery | passed | passed |
| reward-shuffled false selections | 0 | 0 |
| replayed examples | 0 | 0 |

The prior begins with eight source adapters after shift one and 18 after shift
two. Both seeds pass all route, causal, permutation, retention, reversal,
recovery, credit, null-control, and zero-replay gates.

## Claim boundary

This promotes safe reuse of learned external growth state across three shifts
and 32 capabilities. It does not yet promote a reliable sample-efficiency
gain: the prior improves the final-shift floor for seed 69316 but slightly
lowers seed 69317's final-shift floor relative to fresh initialization. Unbounded
growth, learned capacity scheduling, and general continual learning remain
open.

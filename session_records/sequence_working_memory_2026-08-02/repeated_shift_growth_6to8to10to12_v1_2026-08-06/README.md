# Repeated nonstationary growth: length 6 → 8 → 10 → 12 (2026-08-06)

This is the first three-shift expansion audit. Two capabilities are learned on
length-six episodes. Eight new capabilities are acquired from fresh length-
eight episodes, ten more from fresh length-ten episodes, and twelve more from
fresh length-twelve episodes. The controller, earlier route state, and earlier
credit heads remain frozen through all shifts; no earlier examples are replayed.

## Promoted result

| metric | seed 69316 | seed 69317 |
| --- | ---: | ---: |
| phase-1 minimum route selection (length 8) | 0.9219 | 0.8906 |
| phase-2 minimum route selection (length 10) | 0.8906 | 0.9063 |
| phase-3 minimum route selection (length 12) | 0.9219 | 0.8594 |
| old route/permutation | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| credit old / shift 1 / shift 2 / shift 3 / combined | 1 / 1 / 1 / 1 / 1.000 | 1 / 0.875 / 1 / 1 / 0.969 |
| full-bank protection/reversal/recovery | passed | passed |
| reward-shuffled false selections | 0 | 0 |
| replayed examples | 0 | 0 |

Both seeds pass phase-wise route recovery, causal extensions, prior-extension
ordering, all-shift credit, permutation, full-bank protection, isolated
reversal/recovery, and the zero-centered antithetic null. The final bank holds
32 opaque capabilities, crossing the previous closed 20-family bank ceiling.

## Accounting

Per seed: 1,884,424 unique verifier bits, 334,088 unique logical lifetimes,
21,760 optimizer updates, 0 replayed examples, and three distribution shifts.
Across the two promoted seeds: 3,768,848 verifier bits, 668,176 logical
lifetimes, 43,520 optimizer updates, and 0 replayed examples.

## Claim boundary

This promotes three sequential temporal distribution shifts and dynamic
external-bank growth beyond 20 capabilities. It still does not establish
unbounded expansion, learned capacity planning, arbitrary program induction,
or general continual learning: the generated family bank is finite and the
route/credit acquisition machinery remains externally trained.

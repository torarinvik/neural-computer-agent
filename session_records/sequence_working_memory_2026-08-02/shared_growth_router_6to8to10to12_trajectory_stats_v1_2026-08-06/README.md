# Shared candidate growth router: length 6 → 8 → 10 → 12 (2026-08-06)

This audit replaces one extension router per capability with one shared,
permutation-equivariant candidate router per distribution shift. The frozen
controller exposes only learned trajectory statistics: context, final
recurrent state, mean recurrent state, and max recurrent state. Candidate
keys remain opaque random vectors. Three shared routers add 8, 10, and 12
capabilities in sequence, growing the external bank from 2 to 32 rows.

Earlier route state, credit heads, and controller weights remain frozen after
each shift. New routes receive fresh paired verifier outcomes only; no earlier
examples are replayed.

## Promoted result

| metric | seed 69316 | seed 69317 |
| --- | ---: | ---: |
| phase-1 minimum route selection (length 8) | 0.9844 | 0.9375 |
| phase-2 minimum route selection (length 10) | 0.9844 | 0.9688 |
| phase-3 minimum route selection (length 12) | 0.9531 | 0.9375 |
| old route / candidate permutation | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| new candidate permutation | 1.0000 | 1.0000 |
| causal credit across all shifts | passed | passed |
| full-bank protection / reversal / recovery | passed | passed |
| reward-shuffled false selections | 0 | 0 |
| replayed examples | 0 | 0 |

Both seeds pass the hard promotion gates: old-route retention, candidate
permutation invariance, causal new-route recovery, all-shift credit,
full-bank protection, isolated reversal and recovery, reward-shuffled null,
and zero replay. The shared router is a materially more reusable growth
mechanism than the prior one-extension-per-capability design.

## Accounting

Per seed: 4,538,632 unique verifier bits, 105,992 unique logical lifetimes,
104,704 optimizer updates, 0 replayed examples, 3 distribution shifts, and
227.2805/244.9326 wall seconds. Across both seeds: 9,077,264 verifier bits,
211,984 logical lifetimes, 209,408 optimizer updates, 0 replay, and 3 shared
routers per run.

The corrected strict sequential operational route-permutation diagnostic is
`0.9906/0.9911`, matching the direct candidate-score audit. The earlier
`0.4932/0.4943` result was a harness false negative: the permutation audit
compared a remapped physical row to its unpermuted family index and activated
later routers incorrectly. The corrected audit uses the remapped physical
target and is included in both reports.

## Claim boundary

This promotes a shared, variable-bank growth router across three sequential
shifts and 32 generated capabilities. It does not establish unbounded growth,
general program synthesis, broad multimodal transfer, or general continual
learning. The query summary is still a fixed learned representation and route
acquisition uses a large externally supplied update budget. The next pressure
is sample-efficient route acquisition and removal of the fixed trajectory
summary.

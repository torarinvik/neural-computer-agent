# Primary gate measured: bank growth over 64 families, and the wide schema (F83-F84)

## The primary gate (ARCHITECTURE.md §3, settled 2026-08-09)

64 novel families acquired sequentially through ONE frozen plant, every entry
kept, pool 4096, 40000 pre-training updates, 2 seeds.

    amortised_bank.py --pool 4096 --train-updates 40000 --sequential 64

**(a) mastery grows** — 64/64 both seeds; 59/64 and 56/64 by reading alone
(zero gradient steps).

**(b) retention exact** — drift max 0.0, mean 0.0, measured across the whole
grown bank after all 64 entries exist.

**(c) cost vs bank position**

| position | read | acquisition | cold | saving |
| --- | ---: | ---: | ---: | ---: |
| 1-16 | 0.989 | 2.3 | 53.9 | +51.6 |
| 17-32 | 0.998 | 0.8 | 46.1 | +45.3 |
| 33-48 | 0.991 | 9.4 | 50.8 | +41.4 |
| 49-64 | 0.984 | 7.0 | 51.6 | +44.5 |

Overall 4.9 vs cold 50.6 — 10.4x cheaper across 64 acquisitions.

### (c) passes but the test is weak — stated plainly

Entry i+1 is fitted without ever seeing entries 0..i. The plant is frozen and
entries are independent tensors, so nothing here COULD make cost grow with bank
size. Clause (c) as implemented cannot fail, and a gate that cannot fail is not
evidence.

The real (c) needs RETRIEVAL — finding the right entry among N — which this
probe never pays, since entries are handed to the correct family by
construction. That is the missing component. Candidate mechanisms already exist
in this project: F57 cued addressing, F44 consequence probing.

## F84: wide schema at the optimal budget

| | wide @ 20k | wide @ 40k | narrow @ 40k |
| --- | ---: | ---: | ---: |
| toggle | 0.306 | **0.527** | 0.198 |
| perm | 0.708 | 0.986 | 1.000 |
| acquisition | 81.3 | **20.4** | 7.2 |
| cold | 57.9 | 57.9 | 50.0 |

The gate re-crosses (2.8x cheaper). F80's "widening un-crosses the gate" was a
20000-update artefact, exactly as the budget curve predicted.

## Bug caught before it produced a false headline

The first sequential run reported retention drift of -0.43 mean / 0.82 max for
a frozen plant with stored tensors, where the true value is necessarily zero.
`finetune()` optimised a COPY of the entry and returned only (cost, accuracy),
never the tuned tensor, so the sequential arm stored the UNTUNED entry while
comparing against the TUNED accuracy. Clause (b) would have failed loudly and
falsely on its first run. Fixed; drift is now exactly 0.0.

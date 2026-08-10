# Online disjoint-dynamics model routing — promoted

This three-seed rung removes the nested `position + delta` structure used by
the earlier compounding audit. Four regimes share the same opaque state and
intention widths but use four unrelated verifier-private transition tables.
The router receives one opaque transition row at a time, buffers rows without
writing them to the bank, and admits a new context only after factual mismatch
and learned context formation agree.

| metric | seed 70411 | seed 70412 | seed 70413 |
| --- | ---: | ---: | ---: |
| target-C warm/fresh updates | 36 / 44 | 35 / 40 | 27 / 30 |
| target-D warm/fresh updates | 32 / 43 | 31 / 35 | 30 / 33 |
| all-regime planner mastery | 1.0 | 1.0 | 1.0 |
| source-slot mastery after all phases | 1.0 | 1.0 | 1.0 |
| target-C/D admission count | 1 / 1 | 1 / 1 | 1 / 1 |
| target-C/D reuse count | 1 / 1 | 1 / 1 | 1 / 1 |
| old-slot optimizer updates | 0 | 0 | 0 |

Every gate passed in all three seeds: both novel regimes were admitted without labels, later
revisited and routed to their original slots, and all four model slots were
behaviorally mastered. Source slots were byte-stable, wrong-context factual
error rejected the mismatched regime, the controller stayed frozen, and
persistence was exact.

Claim boundary: this promotes disjoint-dynamics routing and model-based
acquisition under a finite opaque transition-table fixture. The context
encoder is pretrained on two source regimes, the transition tables are finite,
and the planner has a finite horizon. It is not yet general continual
learning, unrestricted memory growth, or raw multimodal context discovery.

Reports are protected by `SHA256SUMS`.

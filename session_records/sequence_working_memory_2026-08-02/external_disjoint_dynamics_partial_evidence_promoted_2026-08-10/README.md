# Partial-evidence disjoint-dynamics routing — promoted

This three-seed pressure test gives the online router only a target-covering
subset of each regime's transition rows. The remaining rows are withheld from
the online stream. The partial windows are repeated only to reach the fixed
admission threshold; the router receives no regime label or row-index
metadata. The complete opaque regime sequence is then alternated four times.

| metric | seed 70411 | seed 70412 | seed 70413 |
| --- | ---: | ---: | ---: |
| observed transition rows | 25 / 56 | 25 / 56 | 25 / 56 |
| withheld transition rows | 31 / 56 | 31 / 56 | 31 / 56 |
| target-C warm/fresh updates | 20 / 44 | 14 / 40 | 22 / 30 |
| target-D warm/fresh updates | 20 / 43 | 15 / 35 | 19 / 33 |
| target-C/D reuses after admission | 7 / 7 | 7 / 7 | 7 / 7 |
| all-regime planner mastery | 1.0 | 1.0 | 1.0 |
| source-slot mastery after all phases | 1.0 | 1.0 | 1.0 |
| old-slot optimizer updates | 0 | 0 | 0 |

Every gate passed in every seed. Each novel regime was admitted once and then
routed correctly seven times after admission. All planner probes retained
mastery despite the withheld evidence, source slots stayed byte-stable, the
controller stayed frozen, wrong-context factual error passed, and persistence
was exact.

Claim boundary: this promotes bounded partial-evidence routing where the
withheld rows are selected by a verifier-private fixture so every held-out
planner target remains solvable. It does not establish arbitrary missingness,
noisy multimodal context discovery, unrestricted memory growth, learned
consolidation/compression, or general continual learning.

Reports are protected by `SHA256SUMS`.

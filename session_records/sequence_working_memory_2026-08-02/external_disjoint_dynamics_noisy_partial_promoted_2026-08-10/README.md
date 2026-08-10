# Noisy partial-evidence disjoint-dynamics routing — promoted

This three-seed pressure test combines the two preceding boundaries. Only a
target-covering subset of each regime's transition rows is exposed, with the
remaining rows withheld, and every observed state and next-state tensor is
perturbed with Gaussian noise of standard deviation `0.02`. The resulting
partial windows are alternated for four complete rounds. The router receives
no regime label, row-index metadata, or clean target during admission.

| metric | seed 70411 | seed 70412 | seed 70413 |
| --- | ---: | ---: | ---: |
| observed / withheld transition rows | 25 / 31 | 25 / 31 | 25 / 31 |
| target-C warm/fresh updates | 20 / 44 | 14 / 40 | 20 / 30 |
| target-D warm/fresh updates | 20 / 43 | 15 / 35 | 19 / 33 |
| target-C/D reuses after admission | 7 / 7 | 7 / 7 | 7 / 7 |
| all-regime planner mastery | 1.0 | 1.0 | 1.0 |
| source-slot mastery after all phases | 1.0 | 1.0 | 1.0 |
| old-slot optimizer updates | 0 | 0 | 0 |

Every gate passed in every seed. Both novel regimes were admitted once and
routed correctly seven times after admission. All planner probes retained
mastery, source slots stayed byte-stable, the controller stayed frozen,
wrong-context factual error passed, and persistence was exact.

Claim boundary: this promotes one bounded noisy, target-covering partial
evidence condition over finite opaque transition tables. The missingness mask
is verifier-private and selected so the measured targets remain solvable; the
noise is synthetic and fixed at one standard deviation. It does not establish
arbitrary missingness, real multimodal noise, unrestricted memory growth,
learned consolidation/compression, or general continual learning.

Reports are protected by `SHA256SUMS`.

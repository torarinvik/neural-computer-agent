# Noisy sparse identity and compact consolidation — promoted

This three-seed audit combines arbitrary random half-missingness with noisy
event tensors. Each regime exposes `7/14` transition rows per window, the
observed state and next-state tensors receive Gaussian noise with standard
deviation `0.02`, and the four-regime sequence alternates for four rounds. The
router uses the protected sparse identity index and 16 bounded consolidation
updates from deduplicated external facts.

| metric | seed 70411 | seed 70412 | seed 70413 |
| --- | ---: | ---: | ---: |
| compact fact records after stream | 56 | 56 | 54 |
| target-C warm/fresh updates | 39 / 44 | 39 / 40 | 39 / 30 |
| target-D warm/fresh updates | 39 / 43 | 39 / 35 | 38 / 33 |
| target-C/D reuses after admission | 7 / 7 | 7 / 7 | 7 / 7 |
| all-regime planner mastery | 1.0 | 1.0 | 1.0 |
| source-slot mastery and byte stability | 1.0 / true | 1.0 / true | 1.0 / true |
| sparse identity-retention promotion | true | true | true |

Every seed passes the qualified boundary: the controller remains frozen, both
novel regimes are admitted once, every later revisit is routed correctly, all
planner probes retain mastery, old slots receive zero updates, and persistence
is exact. Compact external-fact reuse is accounted as `632`, `628`, and `623`
rows; raw-row replay remains zero.

Claim boundary: this promotes sparse identity and retention under one
synthetic noise level and random missingness fixture. Uniform sample-efficiency
improvement is not promoted for the latter two seeds. Bootstrap admission is
capacity-guarded, consolidation reads compact external memory, and this does
not establish real multimodal noise, unrestricted memory growth, or general
continual learning.

Reports are protected by `SHA256SUMS`.

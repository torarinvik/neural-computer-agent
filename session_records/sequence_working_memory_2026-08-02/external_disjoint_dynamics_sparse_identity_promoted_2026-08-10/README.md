# Sparse identity and compact consolidation — promoted

This three-seed audit addresses the previously rejected arbitrary-missingness
case. Each regime exposes a deterministic random half-mask (`7/14` rows) on
each window and the complete four-regime sequence alternates for four rounds.
The router uses a compact slot-local evidence index: overlapping factual rows
can recover a known slot, unknown rows remain provisional, and contradictions
block reuse. Sparse reuse is disabled during bootstrap while the bank is still
growing. After each slot has accumulated evidence, the factual model receives
16 bounded consolidation updates from deduplicated external facts. This is
compact external-memory reuse, not replay-free training.

| metric | seed 70411 | seed 70412 | seed 70413 |
| --- | ---: | ---: | ---: |
| observed / withheld rows per window | 7 / 7 | 7 / 7 | 7 / 7 |
| compact fact records after stream | 56 | 56 | 54 |
| target-C warm/fresh updates | 39 / 44 | 38 / 40 | 39 / 30 |
| target-D warm/fresh updates | 39 / 43 | 39 / 35 | 38 / 33 |
| target-C/D reuses after admission | 7 / 7 | 7 / 7 | 7 / 7 |
| all-regime planner mastery | 1.0 | 1.0 | 1.0 |
| source-slot mastery and byte stability | 1.0 / true | 1.0 / true | 1.0 / true |
| sparse identity-retention promotion | true | true | true |

All three seeds pass the qualified promotion boundary: both novel regimes are
admitted once, each is subsequently routed seven times, all regimes retain
planner mastery, source slots remain byte-stable, old slots receive zero
updates, and persistence is exact. This fixes the earlier failure where every
changed sparse subset became a duplicate context and exhausted capacity.

Claim boundary: sparse identity, retention, and compact consolidation are
promoted for this finite opaque transition-table fixture. Sample-efficiency
improvement is not uniformly promoted: seeds `70412` and `70413` are slower
than their fresh controls. The mask is synthetic, bootstrap admission is
capacity-guarded, and compact fact reads are explicitly accounted. This does
not establish arbitrary real multimodal missingness, unrestricted memory
growth, or general continual learning.

Reports are protected by `SHA256SUMS`.

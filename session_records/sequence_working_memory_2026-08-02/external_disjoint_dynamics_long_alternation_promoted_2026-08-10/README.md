# Long alternating disjoint-dynamics routing — promoted

This three-seed pressure test repeats the complete opaque regime sequence four
times after acquisition. Four regimes share one state/intention interface but
use unrelated verifier-private transition tables. The router must continue to
select the original external model slots without labels, old-regime replay, or
controller updates.

| metric | seed 70411 | seed 70412 | seed 70413 |
| --- | ---: | ---: | ---: |
| target-C warm/fresh updates | 36 / 44 | 35 / 40 | 27 / 30 |
| target-D warm/fresh updates | 32 / 43 | 31 / 35 | 30 / 33 |
| target-C reuses after admission | 7 | 7 | 7 |
| target-D reuses after admission | 7 | 7 | 7 |
| all-regime planner mastery | 1.0 | 1.0 | 1.0 |
| source-slot mastery after all phases | 1.0 | 1.0 | 1.0 |
| old-slot optimizer updates | 0 | 0 | 0 |

Every gate passed in every seed. The two novel regimes were admitted once,
then each was routed correctly seven times across the remaining alternating
stream. Source and target slots retained mastery, source slots were
byte-stable, wrong-context factual error rejected the mismatched regime, the
controller stayed frozen, and persistence was exact.

Claim boundary: this promotes longer-horizon routing stability for a bounded
external factual model bank under a finite opaque transition-table fixture. It
does not establish noisy or partial multimodal context discovery, unrestricted
memory growth, learned consolidation/compression, or general continual
learning. The repeated stream adds no model-training replay; it tests memory
selection and retention after acquisition.

Reports are protected by `SHA256SUMS`.

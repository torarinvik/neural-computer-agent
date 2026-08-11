# Sparse factual sidecar seeding — diagnostic rejection

This one-seed diagnostic tested whether initializing the replaceable sparse
factual sidecar from the already-verified source observations would let a
partial continuation reuse a mastered source slot before aggregate model
routing had enough rows.

The hypothesis was rejected at the first diagnostic rung. Seed `70411` kept
the clean control promoted, but the intervention failed every stressed
continuation tested: partial evidence, random missingness, stream noise, and
partial evidence with noise. Source slots remained byte-stable and the
controller remained frozen, so this is a routing/identity failure rather than
catastrophic forgetting. It is recorded as a single-seed diagnostic, not as a
replicated claim.

| condition | promoted | target admission | target reuse | all mastered | sparse retention |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean | yes | yes | yes | yes | yes |
| partial evidence | no | no | no | no | no |
| random missingness | no | no | partial | no | no |
| stream noise `0.005` | no | no | no | no | no |
| partial + noise | no | no | no | no | no |

All five arms used zero old-regime replay during target adaptation and zero
controller updates. The source-side sidecar seed was therefore not sufficient
to make identity robust to incomplete or noisy evidence. The patch was
removed from the canonical experiment rather than retained as an unverified
special case.

The next architectural test is the domain-general expressibility boundary:
measure whether the frozen recipe interpreter can represent a target before
spending search budget, then test a minimal structural basis extension with
matched two-seed interpreter and search-cost controls.

## Provenance

This record is an in-repository diagnostic run from seed `70411`. It does not
import quantitative claims from the exported games session.

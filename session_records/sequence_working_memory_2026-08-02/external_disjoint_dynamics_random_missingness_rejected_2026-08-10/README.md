# Arbitrary missingness disjoint-dynamics routing — rejected

This three-seed audit replaced the verifier-private target-covering mask with
a deterministic random half-mask for each regime and stream position. The
partial windows were alternated for four rounds. The router received no regime
label or row-index metadata, and the held-out planner probes remained clean.

| metric | seed 70411 | seed 70412 | seed 70413 |
| --- | ---: | ---: | ---: |
| observed / withheld rows per regime window | 7 / 7 | 7 / 7 | 7 / 7 |
| target-C reuses after admission | 0 | 0 | 0 |
| target-D reuses after admission | 1 | 0 | 0 |
| target-C mastery after stream | 0.333 | 0.333 | 0.000 |
| target-D mastery after stream | 1.000 | 0.333 | 1.000 |
| target-C/D capacity failures | 7 / 0 | 7 / 7 | 7 / 7 |
| promoted | false | false | false |

The failure is systematic: a new random subset of the same regime is treated
as a novel context because the factual model has not yet learned the withheld
rows. The router then admits or stages duplicate contexts until bank capacity
is exhausted. This is an identity and sparse-evidence routing failure, not a
controller-training failure.

Claim boundary: this rejects the current router for arbitrary missingness. It
does not reject the external factual-memory architecture; target-covering
partial evidence and fixed noisy partial evidence remain separately bounded
promotions. The next fix must make sparse context identity persistent without
allowing an actually novel regime to contaminate an existing slot.

Reports are protected by `SHA256SUMS`.

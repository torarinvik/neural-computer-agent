# Singleton append-stage calibration rejected (2026-08-07)

This control split two outcome-unseen candidates into two isolated stages,
leaving one candidate per stage. Even with 256 fresh updates per stage, both
seeds reach only `0.5000` post-failure routing and stage permutation and
unseen-acquisition gates fail.

The failure is expected from the current pairwise outcome-ranking objective:
a one-candidate stage has no informative within-stage pair. This is retained
as a curriculum and learner-boundary rejection, not as evidence against
multi-stage isolation with minimally learnable candidate groups. The positive
two-candidate-per-stage result is archived separately. Report checksums are in
`SHA256SUMS`.

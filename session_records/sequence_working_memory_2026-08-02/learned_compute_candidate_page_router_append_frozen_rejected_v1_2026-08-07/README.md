# Frozen learned page-router append control rejected (2026-08-07)

This is the direct append-only control for the learned page router. The
source router is trained on three normalized pages covering 30 candidates.
Thirty-four new candidates are then trained in 17 raw external pages while
the source router and source pages remain frozen. No replay or router update
is allowed after append.

The frozen address function does not generalize to unseen pages. Seed 69316
reaches only `0.3958` candidate/page accuracy and seed 69317 only `0.1250`;
most append pages have zero page mastery. Source state, the controller, and
the no-replay accounting controls pass, but candidate, page, and permutation
mastery all fail.

This rejects “append keys and hope the old router generalizes.” The next
intervention is a separately trainable append-router overlay learned only from
new-page verifier outcomes, with verifier-gated fallback from the frozen
source router.

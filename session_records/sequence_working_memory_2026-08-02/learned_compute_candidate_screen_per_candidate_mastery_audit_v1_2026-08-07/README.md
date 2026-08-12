# Per-candidate mastery audit (2026-08-07)

This reruns the factorized append-only bank-20/five-stage boundary with a
strict verifier: aggregate routing is insufficient unless every audited target
also clears `0.75` top-1 mastery. The run adds candidate-key diagnostics so
representation collapse is measured rather than inferred from routing alone.

The replicated result is not promoted. Seed `69316` clears all ten unseen
targets, but seed `69317` has aggregate unseen accuracy `0.8958` while one
unseen target is at `0.0`; it also has known-target holes at `0.7` and `0.0`.
Consequently only one of two seeds passes the strict per-candidate gate.

The key signatures are highly non-separable: nearest-neighbor cosine reaches
`0.9956` and `0.9982`, while effective rank is only `4.47` and `3.59` for
twenty candidates. This makes collapsed upstream signatures, rather than
router capacity, the current bottleneck for reliable growth and consolidation.

Both runs use query-path append priors, five two-candidate stages, 32 fresh
calibration updates per stage, zero replay, and identical frozen-controller
budgets. The reports include the strict per-target metrics, key diagnostics,
all controls, and the required verifier-bit/lifetime accounting.

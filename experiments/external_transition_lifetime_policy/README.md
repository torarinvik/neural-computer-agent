# External transition lifetime policy

This is a short two-seed pressure test for the memory-lifetime boundary.
Each episode creates a fresh three-slot factual transition bank containing
independently trained affine transition models. The bank records usage,
logical age, and factual prediction-error telemetry while those models are
updated and accessed. A hidden verifier identifies which unprotected slot is
safe to remove from the opaque keys and bank-owned telemetry. The learned
policy proposes a stable logical slot ID; the bank's verifier-gated
copy-on-write transaction decides whether the proposal commits.

The evaluation compares the learned policy with matched random and recency
selectors on the same held-out episodes. Retained models must preserve their
held-out transition behavior after eviction. The controller is never
constructed or updated. No old transition rows are retained or replayed:
every episode uses a fresh bank and contributes at most one verifier outcome
to policy adaptation.

This is intentionally a bounded mechanism test, not a claim of general
continual learning. The verifier is synthetic and remains authoritative; the
policy is not yet responsible for consolidation or codec selection. Promotion
requires both seeds to beat random by at least `0.10`, respect the
protected-slot and stable-address gates, preserve retained-model behavior, and
restore exactly.

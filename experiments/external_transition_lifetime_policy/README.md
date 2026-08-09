# External transition lifetime policy

This is a short two-seed pressure test for the new memory-lifetime boundary.
Each episode creates a fresh three-slot factual transition bank. A hidden
verifier identifies which unprotected slot is safe to remove from generic
opaque context, usage, age, and prediction-error features. The learned policy
proposes a stable logical slot ID; the bank's verifier-gated copy-on-write
transaction decides whether the proposal commits. A rejected proposal leaves
the bank unchanged and supplies one outcome bit to the external policy.

The evaluation compares the learned policy with matched random and recency
selectors on the same held-out episodes. The controller is never constructed
or updated. No transition rows are retained or replayed: every episode uses a
fresh bank and contributes at most one verifier outcome to policy adaptation.

This is intentionally a bounded mechanism test, not a claim of general
continual learning. The verifier is synthetic, telemetry is supplied by the
fixture, and the policy is not yet responsible for consolidation or codec
selection. Promotion requires both seeds to beat random by at least `0.10`,
respect the protected-slot and stable-address gates, and restore exactly.

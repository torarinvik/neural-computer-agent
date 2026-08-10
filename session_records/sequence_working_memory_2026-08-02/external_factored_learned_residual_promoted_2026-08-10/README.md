# Promoted external learned factored residuals

This is a bounded three-seed pressure test of context-local factual learning
outside a frozen shared base. The base learned one opaque nonlinear source
dynamics family from `40` pretraining rows and then remained frozen. Two
online regimes shared the state/intention interface; the second had a
different affine residual relation. Each online regime supplied only `16/20`
rows to the external router. The remaining four rows were verifier-private
held-out evidence and were never used for optimizer updates.

The router formed opaque keys from the current evidence bundle, trained an
isolated affine residual-function candidate, and promoted it only after an
independent held-out factual gate and prior-slot retention probe. After both
promotions, source and target bundles alternated six times and routed to
`[0, 1, 0, 1, 0, 1]`. A corrupted bound update was rejected with the committed
model digest unchanged. The controller, shared base, and context encoder were
byte-stable; target adaptation used zero old-regime replay and zero controller
optimizer updates. Exact router persistence passed for every seed.

The learned target held-out MSE was `0.00209`, `0.03312`, and `0.04487`, while
the frozen-base-only control was `0.03903`, `0.38406`, and `0.14780`. A fresh
target model was also measured separately; it won on one seed and lost on two.
This result is therefore a retention/generalization boundary, not a claim of
positive transfer against a fresh learner.

This promotes external learned factual residuals under partial evidence. It
does not establish unrestricted growth, arbitrary nonlinear dynamics,
automatic context formation from raw modalities, compression, or general
continual learning.

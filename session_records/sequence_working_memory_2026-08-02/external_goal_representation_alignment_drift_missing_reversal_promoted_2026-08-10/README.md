# Promoted drift, missing-evidence, and caller-free identity anchors

This four-seed audit composes delayed identity resolution with gradual opaque
signature drift, drift-direction reversals, alternating arrival order, and
masked learned evidence. The bank selected every runtime route and every
identity update from opaque signatures; no runtime frontend or slot ID was
provided to routing or anchor updates.

All seeds `85001`–`85004` passed. Each run routed all 32 identity windows
correctly, exercised 24 partial-evidence windows and 15 order reversals, and
reached `1.0` mastery for both the affine and nonlinear alignments. The
controller, factual model, and verifier memory stayed frozen, persistence was
exact, and replay was zero.

This promotes a bounded replay-free verifier-gated identity-retention result
under gradual/reversible drift and partial evidence. Partial anchors are
intentionally read-only for identity prototypes: storing incomplete vectors is
the next design pressure point. This does not establish semantic open-world
identity discovery, autonomous verifier design, unrestricted memory growth, or
general continual learning.

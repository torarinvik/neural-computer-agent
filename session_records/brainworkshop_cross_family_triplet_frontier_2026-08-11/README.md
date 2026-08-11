# Held-out triplet-parity rule-family frontier (2026-08-11)

This screen extends the cross-family Brain Workshop audit from the fixed
target family to a verifier-private `triplet_parity` target. The protected
prefix contains n-back-2, pair parity, and adjacent switching. The controller,
event encoder, and prefix external files are frozen before target acquisition.

The target file is forced during acquisition so computation and route learning
are measured separately. At 256 target updates, target accuracy was
`63.4--66.8%`; at 512 target updates it rose only to `68.2--73.0%`. The
protected prefix remained perfect and byte-stable, but the target did not
reach the `80%` gate, so held-out route recovery correctly failed.

An additional same-cue curriculum arm first trained the target slot on
`parity2` and then on `triplet_parity` using the same fresh scalar-outcome
protocol. It regressed to `48.6--54.8%`, rejecting naïve mutable-slot reuse as
a compositional strategy.

All arms used zero replay, and controller/encoder state remained unchanged.
The failure localizes the bottleneck to acquiring new computation in the
external capability itself—not routing, retention, or the frozen controller.
The next design should use a generic executable/compositional substrate with
verifier-gated candidate admission.

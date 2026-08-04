# Outcome-only sequential contradiction resolution

This rung tests whether the canonical controller can resolve two contradictory
event streams by updating a latent source-trust belief from its own opaque
action and scalar verifier outcome. One source is privately reliable for a
hidden persistent Markov schedule whose switch hazard is independent of clock
position. Frontends are frozen and the controller never receives the hidden
role, target, or semantic label.

The task is intentionally bounded: the verifier establishes whether adaptive
contradiction resolution is present, not whether the system has broad natural
language or physical-world grounding. Promotion requires three-seed
replication, post-transition adaptation, stream-order invariance, and collapse
when prior outcomes are withheld or shuffled. The current v17 diagnostic is
not promoted; its detailed boundary and lower-learning-rate replication are
recorded under
`session_records/contradiction_amodal_2026-08-03/`.

# Promoted verifier-gated intention admission

Three seeds (`85201`, `85202`, `85203`) pass the copy-on-write intention
admission pressure test. The initial external repertoire contains four
opaque cardinal vectors. `ExternalIntentionCompositionExplorer` proposes the
new vector `[0.5, 0.5]` as the mean of retained entries `(0, 1)`. A held-out
factual verifier admits it; the goal `[1.5, 1.5]` is not mastered before
admission and is mastered afterward through policy-free factual search.

The retained four vectors remain byte-equivalent. A deliberately mismatched
composition candidate (`[1.0, -1.0]`) fails its held-out probe and leaves the
live repertoire unchanged. The controller and factual model are
frozen/unchanged, replay is zero, and exact persistence passes for every seed.
Held-out errors are below `3.3e-14`; rejected-candidate errors are
approximately `0.107`.

This promotes one verifier-gated new-intention transaction whose candidate is
derived from retained opaque experience, not arbitrary intention synthesis,
unrestricted computation, or general continual learning. Composition is
ephemeral and does not mutate the live repertoire; only independently verified
content enters deployed search.

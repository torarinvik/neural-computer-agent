# Promoted verifier-gated intention admission

Three seeds (`85201`, `85202`, `85203`) pass the copy-on-write intention
admission pressure test. The initial external repertoire contains four
opaque cardinal vectors. A held-out factual verifier admits the new vector
`[0.5, 0.5]`; the goal `[1.5, 1.5]` is not mastered before admission and is
mastered afterward through policy-free factual search.

The retained four vectors remain byte-equivalent. A deliberately mismatched
candidate fails its held-out probe and leaves the live repertoire unchanged.
The controller and factual model are frozen/unchanged, replay is zero, and
exact persistence passes for every seed. Held-out errors are below
`3.3e-14`; rejected-candidate errors are approximately `0.015`.

This promotes one verifier-gated new-intention transaction, not arbitrary
intention synthesis, unrestricted computation, or general continual
learning. Candidate content still comes from an external opaque proposer or
experience stream; the new result guarantees that only independently verified
content can enter deployed search.

# Compositional three-procedure external memory

This is the transfer-aware companion to the strict independence rejection.
The bank contains three same-schema artifacts acquired from the same frozen
parent: `complement`, `complement_reverse`, and `complement_rotate`.

The router reached 100% held-out routing, 1/3 under reward shuffling, 100%
under candidate-row permutation, and 1/3 for raw cosine matching. All three
queried procedures had at least one artifact whose execution was causal after
zeroing, the selected artifact was within five percentage points of the best
available artifact for every procedure, and beneficial off-diagonal transfer
was observed for complement ↔ complement-rotate. Bank reload, frozen-core,
and corruption gates passed.

This record deliberately does not claim three independent programs. The
transfer matrix is the result: the complement artifact solves
complement-rotate nearly as well as the dedicated artifact, while the
complement-reverse artifact remains specialized. That is positive evidence
for reusable memory factors and compositional capability, not an address
failure.

Report: `report.json`.

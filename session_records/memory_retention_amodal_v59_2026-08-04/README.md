# v59 address initialization diagnostics

v59 tests whether an identity-initialized learned memory address improves
generalization to unseen event tokens. The long three-seed run preserves
ordinary retention, but unseen-token recall is highly seed-dependent
(`0.547`, `0.719`, `0.996`) and transfer qualifies only two of three seeds.
The short matched rung qualifies transfer in all three seeds but has unstable
main retention. A zero-initialized residual identity path is also negative in
the short diagnostic. These are rejected as permanent branches.

The evidence points to a representation issue rather than a missing identity
initializer: address construction was using transport/age features that differ
between a write and a later recall of the same event. The follow-up v60
corrects that boundary directly by making the canonical address payload-only.

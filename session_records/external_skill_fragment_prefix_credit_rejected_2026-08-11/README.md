# Causal prefix verifier credit — rejected — 2026-08-11

This diagnostic added fresh trainer-only verifier outcomes for every causal
prefix of each ordered fragment program. The serial combiner exposed an
opaque external state snapshot after each boundary, and the same shared action
decoder was trained on those prefix states and the final state. The controller,
register machine, and acquired fragment bank remained frozen during target
learning; operation names and correct actions stayed in the verifier.

## Result

The source-mastered seed-69316 rung used 64 parent updates, 256 updates for
each of four primitive fragments, 128 composition updates, batch size 32,
span 3, audit count 128, and prefix-credit loss weight `0.25`. Source files
were mastered before composition and retained at or above `0.9974`.

| metric | prefix-credit result |
| --- | --- |
| shared training accuracy | `0.5677 / 0.8359 / 0.9271` |
| held-out order accuracy | `0.6042 / 0.4271 / 0.5234` |
| wrong-order accuracy | `0.6354 / 0.8177 / 0.7240` |
| stable shared/fresh bits | none / none |
| unique verifier bits | `1,334,016` |
| prefix-credit verifier bits | `884,736` |
| optimizer updates | `1,472` |
| replayed examples | `0` |
| wall time | `424.47 s` |

The direct prefix objective did not produce stable held-out mastery or
ordered generalization. Compared with the matched final-outcome serial arm,
it did not improve the mean held-out result and made wrong-order rejection
fail. More prefix loss weight is not justified: matched short rungs at weights
`0.25`, `0.5`, and `1.0` also reached no stable prefix.

## Decision

Reject direct prefix decodability as a capability promotion. Retain the
`forward_prefixes()` ABI because it gives verifier-gated trainers causal
execution snapshots without expanding the controller. The result shows that
forcing intermediate states to predict intermediate answers is not the same as
crediting a transition for changing the final behavior.

The next intervention is leave-one-prefix-out causal credit: compare the final
verifier utility with and without one transition under common-random fresh
lifetimes, then train an external transition-use policy from that paired scalar
difference. It must preserve the same frozen-core, no-replay, wrong-order, and
held-out controls.

Claim boundary: this is not general continual learning, unrestricted memory
growth, arbitrary program induction, or compression.

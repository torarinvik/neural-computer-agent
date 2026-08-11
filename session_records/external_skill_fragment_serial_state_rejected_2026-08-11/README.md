# Protected serial external state — 2026-08-11

This diagnostic tests whether the shared-composition bottleneck is caused by
collapsing all fragment evidence into a final residual readout. The new
`ExternalSkillFragmentSerialCombiner` summarizes each opaque fragment segment
and updates an external composition state sequentially. Its state-transition
slots can be appended and protected independently of the controller. The
`serial_shared` arm reuses one transition at every segment, so order must be
carried by state rather than by a position-specific parameter block.

## Result

The source-mastered seed-69316 rung used 64 parent updates, 256 updates for
each of four primitive fragments, 128 composition updates, batch size 32,
span 3, and audit count 128. All primitive source files reached `1.0000`
retention before composition training, and remained at or above `0.9974`
afterward.

| metric | serial-shared result |
| --- | --- |
| shared training accuracy | `0.5286 / 0.8776 / 0.8932` |
| held-out order accuracy | `0.6068 / 0.4453 / 0.5260` |
| wrong-order accuracy | `0.5859 / 0.8568 / 0.6953` |
| stable shared/fresh bits | none / none |
| unique verifier bits | `449,280` |
| optimizer updates | `1,472` |
| replayed examples | `0` |
| wall time | `335.90 s` |

The serial state did not reach stable mastery, held-out transfer remained
below threshold, and wrong-order rejection failed. Missing-evidence,
zero-fragment, reward-shuffled, frozen-parent, frozen-bank, persistence, and
zero-replay controls passed. Because the primitive source was mastered, this
failure is not explained by insufficient atomic-file retention.

## Decision

Reject the serial state as a learned capability promotion. Retain its
versioned, checksummed ABI and protected append-only state mechanism as
optional infrastructure. The result rules out a simple final-readout collapse
as the sole bottleneck: even an explicit state transition does not learn the
ordered execution law from final action outcomes alone.

The next intervention should expose causal prefix execution/credit to the
external learner—while preserving opaque learned traces and scalar verifier
boundaries—rather than adding more position slots, memory capacity, or route
losses.

Claim boundary: this is not arbitrary program induction, unrestricted memory
growth, compression, or general continual learning.

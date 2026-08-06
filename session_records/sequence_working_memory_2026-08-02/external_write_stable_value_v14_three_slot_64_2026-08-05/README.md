# Stable controller value path — seed 17, three-slot 64-update rung

Status: promoted narrow three-slot retention rung; no general continual-learning claim.

The controller now includes an identity-initialized, protocol-agnostic memory
value path keyed by the current learned event and opaque feedback. It is trained
during parent acquisition and frozen during retention; the independently
versioned external writer still learns the write decision and bounded value
adaptation. This removes the context dependence that caused the writer-only
three-slot failure.

Results:

- target-first: `0.963`
- target-last: `0.940`
- target-order gap: `0.022`
- intact: `0.947`
- mastered-parent retention: `0.980`
- unseen-token minimum: `0.945`
- stable bits to threshold: `20,480`
- replayed examples: `0`

The run passed the causal, corruption, clear-memory, parent-retention, and
order-symmetry gates. It is a narrow outcome-only synthetic pressure test, not
evidence for language, physical grounding, arbitrary new computation, or
persistent-memory reload.

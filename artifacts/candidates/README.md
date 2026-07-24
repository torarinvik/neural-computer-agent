# Candidate checkpoints

`core263_parent.pt` is the immutable predictive core selected by the causal
horse race.  It is a candidate rather than a globally promoted checkpoint.

- SHA-256:
  `d027b80a631f61c3a9769b60a079494e0a669e1211d3324a13e5ad7b65a1006d`
- Size: approximately 2.9 MB
- Stable full-task threshold: 48 unique verifier outcomes
- Exact immutable replays: metric-for-metric identical
- Prior-rung compatibility:
  - fixed probe: 16 outcomes
  - fixed target: 48 outcomes
- Frozen-core retention: bit-identical
- Replicated near-transfer:
  - target-side: stable at 8 verifier outcomes
  - observed-effect-side: stable at 8 verifier outcomes
  - effect-target composition: stable at 24 outcomes on two disjoint streams
  - matched fresh initialization: no pass through 64 outcomes

The authoritative behavioral report is:

`experiments/forward_transfer_attention/reports/core263_immutable_exact_controls.json`

This candidate is not yet the complete general neural-computer checkpoint.
Promotion beyond the identify-then-act family requires retention and forward
transfer on additional primitive families.

The near-transfer result localizes to the learned vision encoder and is strong
evidence for reusable visual dynamics, but it remains within a shared rendered
world. The older unrelated spatial and same/different renderers stayed at
chance in the sub-minute screen, so global promotion remains deferred.

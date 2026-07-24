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
- Replicated appearance bridge:
  - novel palette: stable effect-target composition at 24 outcomes
  - novel target/cursor geometry: stable at 24 outcomes
  - palette + geometry: stable at 24 outcomes on two disjoint streams
  - normal/counterfactual accuracy and causal flips: 100% on replication
  - matched fresh and recurrent-only: no pass through 64 outcomes
- Post-promotion retention on a third stream: 8/8/16 outcomes for
  target-side/effect-side/effect-target, with core weights bit-identical
- Cross-attribute color bridge:
  - inherited effect-color atom: stable at 24 outcomes versus 64 fresh
  - reset target-color atom: stable at 64; inherited weights caused negative
    transfer and were correctly discarded for that branch
  - acquired target + effect latents: stable novel relation at 16 outcomes on
    two disjoint streams
  - either atom alone and neither-acquired: no causal pass through 64
  - retained position ladder after promotion: 8/8/24

The assembled color checkpoint is curated separately at
`artifacts/checkpoints/color_primitive_compounder_bits16_seed1901.pt`.

The authoritative behavioral report is:

`experiments/forward_transfer_attention/reports/core263_immutable_exact_controls.json`

This candidate is not yet the complete general neural-computer checkpoint.
Promotion beyond the identify-then-act family requires retention and forward
transfer on additional primitive families.

The transfer result localizes to the learned vision encoder and is strong
evidence for reusable visual dynamics. A separate modular checkpoint now
composes learned target/effect color preferences into a new relation with a
replicated ≥4× experience advantage. Event structure and the binary interface
remain shared, and the older unrelated spatial and delayed same/different
renderers stayed at chance, so global promotion remains deferred.

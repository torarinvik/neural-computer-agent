# Outcome-only source reliability

This rung tests the canonical event-window/controller boundary with frozen raw
frontends. One frontend renders a high bit and three frontends render the same
low bit. The low-bit frontends have hidden source-specific flip rates
`0.05/0.35/0.35`; no confidence or condition label changes when a source is
wrong. The controller receives learned event tokens, generic source keys, and
the scalar verifier outcome only.

The decisive causal audit forces the reliable source `b` to agree against the
two noisy sources `c` and `d`. The reversal flips `b` while leaving `c` and
`d` correct. Across seeds 17, 18, and 19:

- reliable-source conflict reward: `0.9976`, `0.9912`, `0.9966`;
- reversal reward: `0.0015`, `0.0103`, `0.0005`;
- stream-order-shuffled reward: `0.9995`, `0.9883`, `0.9966`;
- all-low-missing reward: `0.5010`, `0.5137`, `0.5088`;
- action-shuffled reward: `0.2451`, `0.2656`, `0.2451`.

Frontends are frozen before training, so the reliability behavior is learned
inside the single controller. A reward-shuffled seed reaches only `0.2729`
clean reward and does not promote. One-missing results are retained as a
diagnostic because two noisy sources make that condition harder than the
promoted conflict primitive.

This promotes narrow outcome-only source reliability over a redundant
synthetic event family. It does not qualify cross-modal relevance, arbitrary
contradiction resolution, or general missing-stream inference.

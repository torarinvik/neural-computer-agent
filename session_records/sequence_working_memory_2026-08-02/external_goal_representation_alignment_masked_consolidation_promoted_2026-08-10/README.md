# Promoted multi-mask identity-memory consolidation

This four-seed audit grows external identity memory from one to four
prototypes per slot, learns three distinct partial observations with different
evidence masks, and then merges two rows through a copy-on-write,
verifier-gated consolidation transaction.

The retention probe preserves the original full route and all three partial
routes. The controller, transition model, verifier statistics, and alignment
adapters remain frozen. Persistence is exact and no old examples are replayed.

Results across seeds `85301`–`85304`:

- affine mastery: `1.0` for every seed;
- nonlinear mastery: `0.9917`–`1.0`;
- slot `0`: four prototypes before consolidation, three after;
- slot `1`: one prototype retained;
- rejected growth and rejected consolidation preserve their source digests;
- all full and differently masked routes survive consolidation and reload;
- replay count: `0`.

The mask union is handled as external memory state and only becomes live after
the retention probe passes. This promotes bounded verifier-gated growth and
compression, not autonomous compression policy, unbounded memory, semantic
open-world identity, or general continual learning.

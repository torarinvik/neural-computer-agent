# Factored external-memory lifecycle promotion

This archive records the five-seed lifecycle pressure test for the canonical
factored router. Each run used a frozen base and a replay-free nonlinear
random-feature residual bank. Two logical regimes were admitted at capacity
`2`; verified growth expanded capacity to `4`; two more regimes were admitted;
storage compression selected `float16_stats`; logical slot `1` was then
evicted under an independent retention probe; and a fifth regime was admitted
as stable logical slot `4`.

Every seed passed all lifecycle gates:

- full capacity blocked a novel third regime before growth;
- growth preserved the first two regimes and changed no model content;
- four regimes routed as `[0, 1, 2, 3]`;
- compressed state round-tripped with held-out retention;
- middle eviction preserved survivor IDs `[0, 2, 3]`;
- the new regime reused available capacity without renumbering survivors;
- final state restored and routed as `[0, 2, 3, 4]`; and
- controller, base, and context encoder digests remained unchanged.

Adaptation used five one-pass sufficient-statistics updates, zero optimizer
updates, and zero old-regime replay per seed. This promotes a bounded,
verifier-gated external-memory lifecycle. It does not establish automatic
open-world context formation, unrestricted memory growth, learned eviction or
compression policy, or general continual learning.

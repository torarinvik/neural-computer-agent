# Explicit modulus — arithmetic-family causal promotion

This two-seed in-repository audit extends the randomized-domain modulus test
to all arithmetic instructions: `INC`, `DEC`, `CINC`, and `CDEC`. Each is
tested on both `m=2` and `m=8`, with domain assignments permuted per random
program. The instruction carries the modulus as an explicit operand; no task
family, semantic label, or privileged answer is provided.

At the stable-prefix threshold `0.9`, all eight correct probes reach stable
execution in all four arms by the 1,500-update rung. The slowest correct probe
is `CINC(m=8)` at update `1,500` in one atomic arm; all other probe thresholds
are earlier. Final correct-probe scores are at least `0.9434`. The
wrong-`m=8` probe on a two-valued target finishes at
`[0.4766, 0.4766, 0.5107, 0.5117]`, near the exact-match chance ceiling.

This promotes a narrow causal learned arithmetic family: the model uses an
explicit value-range operand across unconditional and conditional updates.
It does not promote general continual learning, arbitrary computation,
unrestricted memory growth, or parallel composition as an independently
causal gain. Each arm consumed `192,000` unique random-program steps with zero
replay.

The audit schema is `neural-computer.recipe-expressibility-audit.v3`.

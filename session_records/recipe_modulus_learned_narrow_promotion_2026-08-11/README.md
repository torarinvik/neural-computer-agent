# Explicit modulus — fixed-domain historical promotion

**Status: SUPERSEDED AS CAUSAL EVIDENCE.**

This two-seed in-repository audit tests the corrected arithmetic contract on
mixed slot domains `(2, 2, 8, 8, 8, 8)`. The learner sees only random opaque
programs and explicit modulus-bearing instruction vectors. It receives no
family names, task labels, semantic rules, or privileged answers.

At the stable-prefix threshold `0.9`, the dedicated single-increment probes
for both `m=2` and `m=8` reach stable exact execution in every arm. The latest
threshold is update `1,500` (seed `70422`, atomic-only training, `m=8`); the
other probes reach the threshold earlier. This is a narrow learned arithmetic
promotion with `192,000` unique random-program steps per arm and zero replay.

Because the domain assignment was fixed to slot positions, this result could
not distinguish reading the modulus operand from memorizing slot identity. It
is retained as a historical positive execution result, but the randomized
domain audit is the current causal evidence.

The result does not promote the parallel composition target: it is unstable
in the same run, falling below `0.4` at the final checkpoint on both parallel
arms after earlier higher scores. It also does not promote general continual
learning, arbitrary computation, or unrestricted memory growth.

The deterministic legacy control remains decisive: a global modulus of eight
matches increments at `[0.5, 0.5, 1.0, 1.0, 1.0, 1.0]`, while explicit per-slot
moduli match at all ones. The implementation is in
`src/neural_computer/recipe_basis.py`; the audit schema is
`neural-computer.recipe-expressibility-audit.v2`.

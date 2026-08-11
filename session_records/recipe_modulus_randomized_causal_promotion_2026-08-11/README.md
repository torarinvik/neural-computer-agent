# Explicit modulus — randomized-domain causal promotion

This two-seed in-repository audit removes the fixed-slot confound from the
earlier modulus result. Every random program receives a permutation of the
domain multiset `(2, 2, 8, 8, 8, 8)`, while dedicated probes pin only the
target slot's domain. The instruction carries the modulus as an explicit
operand. No family name, task label, semantic rule, or privileged answer is
provided.

At the stable-prefix threshold `0.9`, both correct single-increment probes
(`m=2` and `m=8`) reach stable exact execution by update `300` in all four
arms: two seeds crossed with atomic-only and parallel-training grammars. The
wrong-`m=8` probe for a two-valued target finishes at
`[0.4766, 0.4766, 0.5107, 0.5117]`, near the expected `0.5` exact-match
ceiling, rather than succeeding through fixed slot identity.

This promotes a narrow causal learned modulus capability. It does not promote
general continual learning, arbitrary computation, unrestricted memory
growth, or parallel composition as an independently causal gain. The parallel
target is stable in the richer arms, but its training-distribution effect is
still a separate confound.

Each arm consumed `192,000` unique random-program steps with zero replay. The
deterministic legacy control remains `[0.5, 0.5, 1.0, 1.0, 1.0, 1.0]` versus
all-one explicit rates. The audit schema is
`neural-computer.recipe-expressibility-audit.v3`.

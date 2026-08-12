# Runtime-generated opaque four-source order permutation

This is the matched source-order control for runtime-generated opaque neural
consolidation. The same five eight-step procedures and budgets were used as the
canonical audit, but sources arrived in order `[4, 3, 2, 0]` instead of
`[0, 2, 3, 4]`. The audit was replicated at seeds `69316` and `69317`.

Both replicas adopted all three shared rewrites and passed source retention,
exact reload, reversal isolation/recovery, corruption rejection, frozen-core,
and zero-replay gates. Reloaded source behaviors were
`0.8164 / 1.0000 / 1.0000 / 1.0000` and `1.0000 / 1.0000 / 1.0000 / 1.0000`.

Retention is therefore order-robust for this bounded test. Transfer efficiency
is not order-invariant: inherited/fresh target stable-prefix budgets were
`8,192 / 6,144` bits at seed `69316` and `4,096 / 6,144` at seed `69317`.
This promotes order-robust retention, not universal program induction or
general continual learning.

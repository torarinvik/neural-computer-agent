# Runtime-generated opaque four-source sequential neural consolidation

This is the grammar-generalization audit for sequential neural consolidation.
Five eight-step procedures were generated at runtime from verifier-private
opaque rules (`--program-seed 4242`, `--primitive-family opaque_rule`). Four
were acquired sequentially into one shared external neural artifact; each
rewrite trained only on the new source and was retention-gated against earlier
aliases. The exact audit was replicated at seeds `69316` and `69317`.

Both replicas adopted all three rewrites, retained all four source aliases at
`1.0000` after reload, passed reversal isolation/recovery and corruption
rejection, kept the controller frozen, and used zero replay. The held-out target
reached `1.0000` after inherited/fresh stable-prefix budgets of `6,144 / 6,144`
bits at seed `69316` and `4,096 / 6,144` at seed `69317`. Thus inherited state
was never worse and was faster in one replica, but this is not a universal
program-induction or general continual-learning claim.

The exact audit consumed 1,439 seconds and 1,088 seconds respectively for
4,736 optimizer updates per seed. Runtime-generated procedure complexity is
therefore a material implementation bottleneck even when the retention gates
pass.

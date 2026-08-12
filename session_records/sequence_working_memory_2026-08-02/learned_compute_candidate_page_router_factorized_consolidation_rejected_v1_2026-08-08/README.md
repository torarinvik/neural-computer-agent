# Factorized shared-router consolidation — rejected

This audit tested replacing three generation-specific routers with one shared
local-page router plus small generation binding state. The first selector used
pooled generation summaries; the second used the full opaque candidate-token
set and 4,096 selector updates. Both retained the original cascade as a
comparison and used zero replayed examples.

The full-token selector reached only `0.679/0.679/0.643` generation accuracy
on the three append generations, even after identity-initialized per-generation
query adapters. The verifier-gated shared-core cascade reached `0.6667`
overall with unresolved rows; page permutation was `0.6667`, and the
reward-shuffled cascade remained a valid null. The local shared router itself
therefore cannot preserve all three bindings under this contract.

The factorized router family is rejected for now. It reduces full-router count
but loses retention. Future consolidation should operate on verified
artifact-level reusable computation or learn a stronger binding mechanism
before attempting another page-router merge.

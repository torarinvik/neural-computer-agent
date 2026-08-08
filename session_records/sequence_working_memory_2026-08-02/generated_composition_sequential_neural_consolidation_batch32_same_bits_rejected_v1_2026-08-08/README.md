# Batch-32 same-verifier-bits consolidation candidate — rejected

This candidate tried to halve optimizer updates by doubling the batch size from
16 to 32 while halving every training budget. The nominal verifier-bit budget
and logical-lifetime accounting were held constant; evaluation frequency was
adjusted to preserve prefix resolution.

The candidate was rejected at the first sequential growth stage. The new
source reached only `0.6953` on one protected alias while the other remained at
`1.0000`, so no shared rewrite was admitted. Parent stability, frozen-core,
reload, corruption, and zero-replay controls passed, but source mastery and
stable-prefix gates did not.

This is negative evidence that larger batches are not a drop-in replacement
for optimizer updates in the current continual-acquisition learner. The next
cost intervention must reduce redundant evaluation/verification or improve
per-update credit assignment, not simply scale batch size.

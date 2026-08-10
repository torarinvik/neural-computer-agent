# Matched-fresh policy-free disjoint compounding — 2026-08-10

This archive records the corrected five-seed audit of the policy-free factual
transition-model boundary. Two source regimes are learned first, followed by
two genuinely disjoint target regimes. A frozen controller emits opaque state
and intention tensors; an external model bank learns factual transitions, and
an inference-only planner derives behavior for the current opaque goal.

Before each target, the bank creates isolated transfer and fresh challenger
models. The fresh challenger is initialized from a caller-owned baseline and
receives the same four-update factual prefix as the transfer candidate. The
matched fresh control trains from the exact unprobed baseline digest. The live
source slot is checked for byte stability, and only the selected candidate is
committed.

| seed | warm cumulative updates | matched fresh updates | transfer choices | result |
| ---: | ---: | ---: | ---: | --- |
| 70411 | 155 | 162 | 0/2 | promoted |
| 70412 | 133 | 145 | 1/2 | promoted |
| 70413 | 125 | 131 | 1/2 | promoted |
| 70414 | 150 | 157 | 0/2 | promoted |
| 70415 | 137 | 145 | 1/2 | promoted |

All seeds mastered both targets, retained every prior model at full planner
mastery with byte-stable source slots, kept the controller frozen, used zero
old-regime replay, passed the verifier-only no-agent floor, and preserved
exact persistence. The reports explicitly record the matched fresh digest.

This promotes a fair, reproducible challenger protocol and a five-seed
policy-free disjoint compounding signal. It remains bounded evidence: the
context encoder, synthetic dynamics family, finite model capacity, fixed
four-update probe, and finite planner horizon do not establish unrestricted
memory growth or general continual learning.

Reports are protected by `SHA256SUMS`.

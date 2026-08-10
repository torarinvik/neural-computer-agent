# Long alternating nonlinear memory lifecycle

Three seeds (`82301`, `82302`, `82303`) passed a 16-round lifecycle pressure
test. Six nonlinear regimes were acquired through two verified capacity
expansions (`4 -> 6 -> 7`). An equivalent context was consolidated, reducing
physical models from seven to six; a later update detached that context
copy-on-write using 64 fresh target-regime rows while preserving the source
model byte-for-byte.

Before and after detachment, all six regimes retained their held-out factual
errors across 16 complete alternation rounds. The selected statistics-aware
`float16_stats` representation preserved the same 96 visits per seed after
compression and restore. Logical addresses, historical model digests, router
state, and corruption rejection all remained exact. The controller stayed
frozen, replayed examples were zero, and consolidation used zero optimizer
updates. Each seed accounted for 896 unique verifier bits and 832 logical
lifetime observations.

This promotes bounded long-alternation storage lifecycle safety. It does not
establish semantic merging, unrestricted memory growth, or general continual
learning.

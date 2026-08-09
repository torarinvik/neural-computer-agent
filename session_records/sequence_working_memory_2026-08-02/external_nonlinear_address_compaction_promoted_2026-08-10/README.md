# Nonlinear address compaction and codec safety — promoted

This three-seed lifecycle audit composes learned copy-on-write addresses with
verifier-gated factual-memory growth and compaction. Each run grows capacity
`4 -> 6 -> 7`, creates an equivalent copy-on-write target slot, consolidates
the two equivalent parameter sets while preserving both opaque logical
addresses, and round-trips the alias relationship through the storage
payload.

| seed | physical models | max legacy delta | max stats-f16 delta | selected codec |
| ---: | ---: | ---: | ---: | --- |
| 82301 | 7 -> 6 | 1.1233 | 4.2e-5 | `float16_stats` |
| 82302 | 7 -> 6 | 6.5146 | 2.1e-5 | `float16_stats` |
| 82303 | 7 -> 6 | 5.0306 | 7.9e-6 | `float16_stats` |

All promoted gates passed: both growth transactions, target acquisition,
equivalent-slot consolidation, stable logical addresses and historical model
digests, alias persistence, corruption rejection without a bank write, frozen
controller, zero replay, and exact router persistence.

The legacy per-tensor float16/int8 and row-int8 codecs were correctly rejected
for the replay-free random-feature sufficient-statistics family. The promoted
`float16_stats` codec preserves the basis and normal matrix, quantizes the
solved predictor, and reconstructs the target matrix on restore. It reduced
storage from `487,564` to `483,952` bytes per physical model bank while all
held-out deltas stayed below `5e-5`.

Claim boundary: bounded retention-verified factual-memory lifecycle and
statistics-aware storage compression. This does not establish semantic
merging, unrestricted memory growth, or general continual learning.

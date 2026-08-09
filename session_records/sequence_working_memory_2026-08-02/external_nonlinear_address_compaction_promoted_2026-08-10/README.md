# Nonlinear address compaction and codec safety — promoted

This three-seed lifecycle audit composes learned copy-on-write addresses with
verifier-gated factual-memory growth and compaction. Each run grows capacity
`4 -> 6 -> 7`, creates an equivalent copy-on-write target slot, consolidates
the two equivalent parameter sets while preserving both opaque logical
addresses, and round-trips the alias relationship through the storage
payload.

| seed | physical models | max float16 delta | max int8 delta | codec decision |
| ---: | ---: | ---: | ---: | --- |
| 82301 | 7 -> 6 | 1.1233 | 0.7218 | reject both |
| 82302 | 7 -> 6 | 0.6080 | 6.5146 | reject both |
| 82303 | 7 -> 6 | 0.0951 | 5.0306 | reject both |

All promoted gates passed: both growth transactions, target acquisition,
equivalent-slot consolidation, stable logical addresses and historical model
digests, alias persistence, corruption rejection without a bank write, frozen
controller, zero replay, and exact router persistence.

The current float16/int8 codecs were correctly not promoted for the
replay-free random-feature sufficient-statistics family. Quantizing its normal
equation matrices causes held-out factual drift beyond the `1e-3` baseline
delta. This is a positive safety result, not evidence that compression works;
the next implementation is a statistics-aware codec with the same verifier.

Claim boundary: bounded retention-verified factual-memory lifecycle and safe
codec rejection. This does not establish semantic merging, unrestricted
memory growth, or general continual learning.

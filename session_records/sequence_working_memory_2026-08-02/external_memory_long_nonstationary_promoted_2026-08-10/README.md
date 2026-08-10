# Long nonstationary external-memory maintenance

This archive records the first audit in which one external transition bank
survives a long nonstationary stream rather than being reset for each
maintenance scenario. The memory-side policy repeatedly chooses among
`grow`, `share`, `compress`, `evict`, and `defer` from generic storage
telemetry. Each committed mutation is an actual copy-on-write,
retention-verified bank transaction.

Seeds `6120`, `6121`, and `6122` all pass the promoted gates. Across the
seeds, the persistent bank performs two growth transactions, one factual
sharing transaction, four compression transactions, and two safe eviction
transactions while acquiring four recurring opaque capabilities. The minimum
retention floor is at least `0.9991`, compression saves `720` bytes, and no
novel event is missed by the trained stream.

The trained policy's mean online utility is `0.9953`, `0.9969`, and `0.9969`.
Matched shuffled-verifier controls reach only `0.5156`, `0.5328`, and
`0.5188`, showing that the rapid maintenance compounding depends on the
verifier outcomes. The action-shuffled controls are a weaker but positive
causal control: their online utility remains close because repeated future
opportunities eventually repair random choices, so final held-out utility is
reported separately and is not used as the sole control.

The stable online threshold is reached at `64` unique verifier bits on every
seed. Mean maintenance latency is `2.06–2.16 ms` and p95 latency is
`2.57–2.76 ms` on the local audit runtime. A transfer ratio against the fresh
policy is intentionally reported as undefined because the fresh policy's
online utility is exactly zero; the absolute gain is the valid comparison.

This promotes bounded replay-free repeated maintenance with real growth,
sharing, compression, and eviction receipts. It does not establish
unrestricted memory growth, learned verifier design, autonomous candidate
equivalence discovery, arbitrary new computation, or general continual
learning.

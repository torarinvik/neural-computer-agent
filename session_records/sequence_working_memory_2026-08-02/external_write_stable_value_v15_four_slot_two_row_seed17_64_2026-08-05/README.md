# Stable controller value path — four-slot, two-row rung

Status: promoted narrow four-slot retention rung.

The promoted three-slot mechanism was tested with a four-slot temporal prefix
and two durable memory rows.

- target-first: `0.981`
- target-last: `0.982`
- intact: `0.986`
- mastered-parent retention: `0.992`
- unseen-token minimum: `0.980`
- stable bits to threshold: `25,600`
- replayed examples: `0`

All causal, corruption, clear-memory, parent-retention, and order-symmetry
gates passed. Persistence is covered by the independent replication run.

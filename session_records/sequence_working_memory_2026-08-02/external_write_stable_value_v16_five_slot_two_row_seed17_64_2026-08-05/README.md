# Stable controller value path — five-slot, two-row seed 17

Status: promoted narrow five-slot retention rung.

The four-slot mechanism was extended to a five-slot temporal prefix and two
durable memory rows.

- target-first: `0.974`
- target-last: `0.973`
- intact: `0.975`
- mastered-parent retention: `0.961`
- unseen-token minimum: `0.965`
- stable bits to threshold: `30,720`
- replayed examples: `0`

The causal, corruption, clear-memory, missing-evidence, parent-retention, and
order-symmetry gates passed. Persistence is covered by the independent seed
69415 replication audit.

# Joint learned binding and factual-memory lifecycle

This archive records the next promoted boundary after outcome-trained
anonymous binding proposals: a learned proposal is now coupled atomically to
held-out factual-memory retention. The external transaction operates on
copy-on-write binding and multi-stream router state, consumes provisional
evidence once with streaming affine sufficient statistics, checks an
independent held-out transition, and commits both replacements only after the
scalar verifier authorizes them.

Across seeds `2501` and `2502`:

- five anonymous streams were represented by two live and three delayed
  provisional identities;
- both learned policies selected the correct provisional/live replacement;
- scalar rejection and wrong-held-out rejection were atomic;
- the retained sibling factual slot survived and the new slot routed;
- a matched-identity drift control returned factual `conflict` without changing
  the factual-bank content digest;
- exact persistence, frozen controller, frozen binding encoder, and zero-replay
  gates passed;
- each seed used 483 scalar verifier bits, 280 identity optimizer updates,
  480 policy optimizer updates, zero factual optimizer updates, and zero
  controller optimizer updates.

The raw reports are `report_seed2501.json` and `report_seed2502.json`.
`sample_efficiency_ledger.json` records the replicated accounting and
`SHA256SUMS` protects this archive.

## Claim boundary

This promotes a bounded joint learned binding/factual replacement transaction
under a held-out gate. It does not establish learned verifier design,
autonomous eviction economics, unrestricted factual-memory growth, arbitrary
drift recovery, or general continual learning.

# Concurrent stream robustness under bounded factual memory

This two-seed pressure test extends the shared factual-bank boundary with
missing evidence, contradictory evidence, drift, and verifier-gated eviction.
Two opaque stream bindings share one capacity-two factual bank.

## Result

Stream 0 continued while stream 1 was missing one row; stream 1 retained a
bounded pending window and later resumed correctly. A contradictory stream-1
bundle returned `pending` then `conflict`, left the committed bank unchanged,
and staged no candidate. After a retention-verified eviction of stream 1's old
slot, its drifted factual model was staged and promoted into the freed slot.
Stream 0 retained its original model digest and routed correctly after the
replacement. Persistence restored both bindings, and checksum corruption was
rejected.

| seed | missing isolated | contradiction safe | evicted | drift promoted | retention | reload |
| ---: | --- | --- | ---: | --- | --- | --- |
| 2201 | pass | pass | 1 | pass | pass | pass |
| 2202 | pass | pass | 1 | pass | pass | pass |

Both runs used a frozen controller, zero optimizer updates, and zero replayed
examples. The stream keys are caller-owned opaque binding tokens, not task
labels.

## Claim boundary

This promotes bounded missing/contradictory/drifting stream handling over one
shared factual bank with retention-safe replacement. It does not establish
learned identity formation, unrestricted memory growth, arbitrary computation,
or general continual learning. The next rung is learned stream identity and
delay/reliability adaptation under genuinely asynchronous event arrival.

Full accounting is in `sample_efficiency_ledger.json`; raw reports are in
`report_seed2201.json` and `report_seed2202.json`.

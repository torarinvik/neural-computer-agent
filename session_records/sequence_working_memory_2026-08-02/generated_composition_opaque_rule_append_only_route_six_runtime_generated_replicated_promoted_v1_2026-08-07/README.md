# Six-row runtime-generated append-only routing (2026-08-07)

Status: replicated promoted bounded address-discovery result.

This audit moves beyond direct alias lookup. Six runtime-generated opaque
procedures were acquired into six independently protected physical artifact
rows. An append-only route chain then learned to address newly appended rows
without modifying earlier route slices or artifact weights.

| metric | seed 69316 | seed 69317 |
| --- | ---: | ---: |
| protected physical rows | 6 | 6 |
| artifact behavior range | 0.8945–1.0000 | 0.9492–1.0000 |
| learned route accuracy | 1.0000 | 1.0000 |
| key-permutation accuracy | 1.0000 | 1.0000 |
| reloaded route accuracy | 1.0000 | 1.0000 |
| reward-shuffled route accuracy | 0.0000 | 0.0000 |
| frozen core | unchanged | unchanged |
| replayed examples | 0 | 0 |

Each replica used 337,920 unique verifier bits, 93,696 logical lifetimes,
1,664 artifact optimizer updates, and 2,560 route optimizer updates. Wall
time was approximately 671s and 654s.

The six-row short control was safely rejected before route training because
the capacity planner would have needed to evict an unprotected row. The
trainer now records that failure with exact partial accounting instead of
crashing; its report has route_training_started=false, replayed_examples=0,
and an explicit protection mask.

This promotes learned routing across six growing physical rows over a finite
opaque-rule family. It does not establish unbounded address discovery,
learned compression, arbitrary new computation, or general continual
learning.

Evidence files:

- report_seed69316.json
- report_seed69317.json
- report_short_rejected_seed69316.json

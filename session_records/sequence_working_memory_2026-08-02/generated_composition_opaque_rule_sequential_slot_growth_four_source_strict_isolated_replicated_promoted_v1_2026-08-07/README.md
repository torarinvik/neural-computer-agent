# Strict isolated-slot opaque-operator growth to four procedures (2026-08-07)

Status: replicated promoted bounded continual-memory result.

This audit extends the strict isolated-slot protocol from three to four
verifier-private procedures. Each new procedure is trained in a fresh external
slot, then appended under an opaque alias into the existing physical artifact
row. Earlier slot parameters, decoders, and alias bindings are not updated.
The fifth procedure is admitted afterward into a newly grown physical row.

The procedure family contains five eight-step programs over the 256 opaque
three-cell local rules. The deployed learner receives only rendered event
streams and scalar verifier outcomes; rule tables, program IDs, and target
labels remain verifier-private.

| metric | seed 69316 | seed 69317 |
| --- | ---: | ---: |
| staged additions adopted | `3/3` | `3/3` |
| source reload behavior | `1.0000/0.9961/1.0000/0.9766` | `1.0000/0.9648/1.0000/0.9844` |
| target reload behavior | `1.0000` | `1.0000` |
| target physical-row admission | `1 -> 2` | `1 -> 2` |
| retention observations | `64` | `64` |
| controller digest | unchanged | unchanged |
| replayed examples | `0` | `0` |

Each replica used `151,552` unique verifier bits, `51,200` logical
lifetimes, and `1,664` optimizer updates. Wall time was approximately `197s`
per seed. Stable source acquisition required at most `8,192` verifier bits;
the newly grown target required `2,048` stable bits in both seeds.

The short `32/64/64` curriculum control was rejected before promotion because
source mastery and stable retention were not reached. A separate 1,024-update
fresh shared-consolidation control was also rejected: it reduced payload size,
but the third source retained only about `0.63` across probes. This confirms
that the reliable mechanism is append-only isolated capacity, not longer
training of one shared consolidator.

This promotes four-procedure replay-free acquisition and retention in bounded
external memory. It does not establish arbitrary program induction,
unrestricted memory growth, learned compression, or general continual
learning.

Evidence files:

- `report_seed69316.json`
- `report_seed69317.json`
- `report_short_control_seed69316.json`
- `report_shared_consolidation_1024_rejected_seed69316.json`

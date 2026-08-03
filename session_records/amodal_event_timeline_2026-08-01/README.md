# Timestamp-preserving asynchronous input — 2026-08-01

## Result

The synchronous complementary N=2 bus now survives transport-level
asynchrony. `AmodalEventTimeline` sorts out-of-order encoded events by their
generic timestamps and groups only events within a declared tolerance.

At 4,096 held-out pair-relation lifetimes using the frozen controller and
promoted input bus:

| Delivery | Accuracy | Equality with synchronous actions |
| --- | ---: | --- |
| One partial stream | 55.57% | — |
| Synchronous N=2 | 96.36% | reference |
| Out-of-order, preserved timestamps | 96.36% | exact |
| 0.25 timestamp jitter, tolerance 0.5 | 96.36% | exact |

Two events carrying genuinely mismatched timestamps produced two separate
windows. The transport layer therefore does not silently merge stale evidence.

## Claim boundary

This is timestamp-aware transport alignment, not a learned delay policy. The
controller never receives arrival order, stream names, or task metadata; the
timeline only uses the generic timestamp field. Learned decisions about how
long to wait, whether to retrieve stale events, and how to trade latency for
additional evidence remain future experiments.

The audit artifact is `timestamp_timeline_audit_4096.json` in this directory.
The implementation is
`experiments/archive/unified_cognitive_controller/audit_amodal_event_timeline.py`.

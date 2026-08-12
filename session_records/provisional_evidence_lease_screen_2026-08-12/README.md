# Bounded provisional evidence lease screen

Date: 2026-08-12

This screen tests one external-memory continuity lease against the current
replay-free online transition-discovery baseline. The lease is allowed only
when one provisional candidate has filled the configured bank capacity. It
routes one additional opaque evidence bundle to that isolated candidate, then
expires. It does not update the controller, committed source slot, or any
device/protocol output and does not weaken promotion gates.

The canonical baseline is the post-fallback `last_token` context address with
masked mean/max event-window statistics, `window_gain=0.05`, routing tolerance
`0.01`, seeds `80–103`, and one `active_interleaved` probe. The lease arm uses
the same configuration with `provisional_evidence_lease_bundles=1`.

## Results

| mode | complete | promotions | fresh improvements | controller unchanged | source stable | replayed rows | transition rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline active-interleaved | 62/96 | 68/96 | 62/96 | 96/96 | 96/96 | 0 | 2,868 |
| lease active-interleaved | 63/96 | 71/96 | 63/96 | 96/96 | 96/96 | 0 | 2,868 |
| baseline post-training active | 58/96 | 62/96 | 58/96 | 96/96 | 96/96 | 0 | 2,868 |
| lease post-training active | 57/96 | 66/96 | 57/96 | 96/96 | 96/96 | 0 | 2,868 |
| lease passive | 51/96 | 60/96 | 51/96 | 96/96 | 96/96 | 0 | 2,868 |

The active-interleaved improvement is therefore real but schedule-dependent.
The lease is retained as a bounded opt-in policy and rejected as a universal
default. It remains a memory/evidence-flow result, not evidence of arbitrary
new computation, unrestricted memory growth, or general continual learning.

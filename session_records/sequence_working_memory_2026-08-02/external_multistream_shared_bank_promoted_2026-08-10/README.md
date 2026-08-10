# Shared factual bank with interleaved stream bindings

This two-seed sub-minute pressure test validates the new
`ExternalMultiStreamTransitionContextRouter` boundary. Three opaque stream
bindings were interleaved while using one shared factual model bank and one
frozen controller boundary.

## Result

Both seeds promoted all three stream-local factual candidates. Updating stream
0 left streams 1 and 2's provisional candidate digests unchanged. After
one-pass factual adaptation and held-out verification, revisits routed to
stable slot IDs `[0, 1, 2, 0, 1, 2]`; persistence restored the same bindings,
and checksum corruption was rejected.

| seed | streams | promotions | routing | persistence | corruption rejected |
| ---: | ---: | ---: | --- | --- | --- |
| 1901 | 3 | 3/3 | 0,1,2,0,1,2 | pass | pass |
| 1902 | 3 | 3/3 | 0,1,2,0,1,2 | pass | pass |

The controller was frozen, optimizer updates were zero, and replayed examples
were zero. The affine sufficient-statistics models consumed each training row
once; the three held-out rows were separate promotion evidence.

## Claim boundary

This promotes a bounded shared-bank stream-binding and persistence invariant.
The stream key is an opaque caller-owned binding token, not a task label, so
this does not establish learned identity formation, arbitrary computation,
unrestricted memory growth, or general continual learning. The next pressure
test is concurrent streams with missing, contradictory, and drifting evidence
under bounded eviction.

Full accounting is in `sample_efficiency_ledger.json`; raw reports are in
`report_seed1901.json` and `report_seed1902.json`.

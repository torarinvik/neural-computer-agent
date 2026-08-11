# Failure-driven external route reversal

This promotion tests whether an isolated external compute file can be retained
while its route is changed from fresh scalar outcomes when the task behind a
rendered cue reverses.

## Protocol

- The frozen controller and event frontend produce the route key.
- `symbol_parity` is first acquired behind cue `7` and protected.
- `triplet_parity` is acquired as a second opaque file behind cue `8`.
- The verifier then changes the task behind cue `7` to `triplet_parity`.
- Both opaque files are probed in parallel; only terminal scalar episode
  outcomes are observed by route memory.
- Four consecutive failure-driven observations demote the stale route and
  promote the replacement for cue `7`.
- The original file is evaluated directly afterward to test retention, and cue
  `9` tests that an unseen context does not simply select the replacement.

## Results

Seeds `17` and `18` both passed every promotion gate:

| Measure | Seed 17 | Seed 18 |
| --- | ---: | ---: |
| Changed same-cue accuracy | 0.8750 | 1.0000 |
| Old-file forced retention | 1.0000 | 1.0000 |
| Unknown-cue accuracy | 0.5099 | 0.5107 |
| Stale-route reversal count | 1 | 1 |
| Replayed examples | 0 | 0 |

Both runs also reproduced the changed route exactly after serialization and
reload. The controller, event encoder, old file, and replacement file were
byte-identical across the reversal. Each seed used `274,048` training verifier
bits, `10,624` audit verifier bits, `23,168` logical lifetimes, `448` optimizer
updates, and `276` route-memory updates.

## Claim boundary

This is a replicated, bounded nonstationary-memory result: failure-driven
same-context replacement with retention of the old computation. It does not
yet establish unrestricted memory growth, arbitrary new computation, semantic
ambiguity resolution, or general continual learning without catastrophic
forgetting.

The raw reports are `seed-17.json` and `seed-18.json`. The implementation is
`experiments/brainworkshop_canonical/external_compute_route_reversal.py`.

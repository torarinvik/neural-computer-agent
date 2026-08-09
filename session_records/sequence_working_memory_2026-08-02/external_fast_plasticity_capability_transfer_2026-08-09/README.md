# External fast-weight capability-adapter transfer

Date: 2026-08-09; seeds: `69316`, `69317`

This audit tests whether a learned memory-side intention adapter transfers to
a new capability when the controller and inherited program are frozen. The
source adapter sees `64` unique opaque action/value lifetimes. The target gets
fresh external fast-weight state and `16` unique positive outcome lifetimes.
The matched fresh control receives the same target stream and trains a new
adapter online. No source examples are replayed.

## Result

| seed | inherited stable examples | fresh-control stable examples | inherited target floor | source-retention floor |
| ---: | ---: | ---: | ---: | ---: |
| 69316 | 1 | 7 | 0.9971 | 0.9971 |
| 69317 | 1 | 14 | 0.9979 | 0.9979 |

Both seeds passed the promotion gate. The inherited program made zero target
optimizer updates, old source states remained protected, and failed outcomes,
missing evidence, exact persistence, and frozen-parameter controls passed.
Accounting was `80` unique verifier bits and `80` unique logical lifetimes per
seed, with `0` replayed examples. The fresh-control stream is a paired
comparison, not additional unique experience for the inherited learner.

## Claim boundary

This is a qualified interface-prior transfer result: a shared learned
action-to-intention adapter makes a new isolated memory state immediately
usable. It is not evidence for general continual learning, unrestricted memory
growth, arbitrary new computation, multi-step credit assignment, or learned
compression/eviction. The tested relation is intentionally regular and the
next audit must vary it and require a genuinely new multi-step capability.

Detailed per-seed summaries are in `report_seed69316.json` and
`report_seed69317.json`. The reproducer is
`experiments/external_fast_plasticity/train_capability_adapter.py`.

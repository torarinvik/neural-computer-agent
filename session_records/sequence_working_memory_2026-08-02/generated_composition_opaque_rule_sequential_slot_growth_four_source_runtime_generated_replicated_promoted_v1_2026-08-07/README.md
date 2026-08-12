# Runtime-generated four-source isolated growth (2026-08-07)

Status: replicated promoted bounded continual-memory result.

This is the runtime-generated version of the strict four-source isolated-slot
audit. The verifier sampled five distinct eight-step procedures from the
256-member opaque three-cell local-rule family with `program_seed=4242`. Four
were acquired sequentially in fresh external slots; the fifth was admitted
afterward into a newly grown physical row.

The program tuples and rule tables remain verifier-private. The deployed
learner received only rendered event streams and deterministic scalar outcomes;
no program ID, rule token, task label, or correct action entered the controller.

| metric | seed 69316 | seed 69317 |
| --- | ---: | ---: |
| staged additions adopted | `3/3` | `3/3` |
| source reload behavior | `1.0000/0.9961/1.0000/0.9766` | `1.0000/0.9648/1.0000/0.9844` |
| target reload behavior | `0.8750` | `1.0000` |
| target physical-row admission | `1 -> 2` | `1 -> 2` |
| retention observations | `64` | `64` |
| controller digest | unchanged | unchanged |
| replayed examples | `0` | `0` |

Each replica used `151,552` unique verifier bits, `51,200` logical
lifetimes, and `1,664` optimizer updates. Wall time was approximately `203s`
per seed. Stable source acquisition required at most `8,192` verifier bits;
the target required `8,192` and `4,096` fresh stable bits respectively.

The short `32/64/64` control was rejected before promotion because source
mastery and stable retention were not reached. All integrity, reversal,
corruption, frozen-core, exact-reload, and zero-replay gates passed on both
promoted replicas.

This closes the manual-program-specification gap for four-source isolated
growth. It still proves only bounded external capacity over a finite opaque
operator family; arbitrary program induction, learned compression,
unrestricted memory growth, and general continual learning remain open.

Evidence files:

- `report_seed69316.json`
- `report_seed69317.json`
- `report_short_control_seed69316.json`

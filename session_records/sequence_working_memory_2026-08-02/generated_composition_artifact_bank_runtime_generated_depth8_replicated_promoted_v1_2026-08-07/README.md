# Runtime-generated eight-step program growth (2026-08-07)

Status: replicated promoted deeper-computation result.

The generated-composition renderer and verifier-private grammar now support
up to eight ordered primitives. Each ordinal band binds a primitive cue to an
opaque position marker in the event image; the controller receives no program
tuple, primitive name, or composition ID. The frozen controller, external
artifact blueprint, and append-only route chain remain unchanged.

The runtime schedule generated from seed `2718` was:

1. `forward -> prefix_parity -> reverse -> forward -> global_parity -> global_parity -> rotate -> complement`
2. `global_parity -> reverse -> reverse -> global_parity -> reverse -> reverse -> global_parity -> prefix_parity`
3. `rotate -> forward -> forward -> complement -> forward -> reverse -> adjacent_xor -> prefix_parity`

| metric | seed 69316 | seed 69317 |
| --- | ---: | ---: |
| artifact behavior | `1.0000/1.0000/0.8711` | `1.0000/1.0000/0.9805` |
| route accuracy | `1.0000` | `1.0000` |
| candidate permutation | `1.0000` | `1.0000` |
| cold-start old-route retention | `1.0000` | `1.0000` |
| reward-shuffled route control | passed | passed |
| reload/corruption/frozen-core/zero-replay | passed | passed |

All three rows were stably mastered and protected in both replicas. Each seed
consumed `162,816` unique verifier bits, `43,776` unique logical lifetimes,
`896` artifact optimizer updates, and `1,024` route updates. Wall time was
`307.7s` and `306.1s`.

The short (`8/16/32`) and medium (`32/64/128`) controls retained route and
permutation gates but failed artifact mastery at `0.6250` and `0.5625` for
the weakest generated program. They are retained as acquisition-depth
controls; the mastery threshold was not weakened.

This promotes an eight-step runtime-generated computational interface and
shows that deeper ordered procedures can be acquired without changing the
controller. It remains bounded: the primitive registry, eight-step renderer,
artifact blueprint, and append-only capacity are finite. Open-ended program
induction, learned compression, unrestricted memory growth, and general
continual learning remain unqualified.

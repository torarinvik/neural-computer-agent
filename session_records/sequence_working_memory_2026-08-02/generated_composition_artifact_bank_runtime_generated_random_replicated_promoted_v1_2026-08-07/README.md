# Runtime-generated compositional artifact growth (2026-08-07)

Status: replicated promoted mechanism-transfer result.

This audit generates a fresh verifier-private program schedule at runtime
from seed `1739`. The schedule is not selected from the fixed composition
table, and generated programs are rejected when their function duplicates an
existing default program. The frozen controller, external artifact blueprint,
and append-only route chain are unchanged. Each program is learned in a new
isolated artifact from fresh verifier outcomes, admitted only after stable
retention, and routed through opaque candidate keys.

The generated schedule was:

1. `reverse -> complement -> rotate -> global_parity`
2. `reverse -> global_parity -> reverse -> adjacent_xor`
3. `prefix_parity -> prefix_parity -> rotate -> complement`

| metric | seed 69316 | seed 69317 |
| --- | ---: | ---: |
| artifact behavior | `1.0000/1.0000/0.8828` | `1.0000/1.0000/0.8906` |
| route accuracy | `1.0000` | `1.0000` |
| candidate permutation | `1.0000` | `1.0000` |
| cold-start old-route retention | `1.0000` | `1.0000` |
| reward-shuffled route control | passed | passed |
| reload/corruption/frozen-core/zero-replay | passed | passed |

Both runs protected all three rows, reloaded exact route behavior, rejected
artifact corruption, left the parent core unchanged, and used zero replay.
Each seed consumed `162,816` unique verifier bits, `43,776` unique logical
lifetime units, `896` artifact optimizer updates, and `1,024` route updates.
Wall time was `430.2s` and `431.5s`.

The short (`8/16/32`) and medium (`32/64/128`) curriculum controls both
correctly refused the first append because the first generated artifact had
not reached protection. Those rejections are retained as evidence that the
growth path does not silently evict an unmastered capability.

This promotes runtime-generated mechanism transfer beyond a predeclared
append schedule. It remains bounded external growth: arbitrary open-ended
program induction, learned compression, unrestricted memory growth, and
general continual learning remain unqualified.

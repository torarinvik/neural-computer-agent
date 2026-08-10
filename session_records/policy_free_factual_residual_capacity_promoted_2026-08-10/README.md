# Promoted: capacity-scaled factual residual memory

Date: 2026-08-10
Seeds: `101`, `102`
Schema: `neural-computer.policy-free-factual-residual-capacity.v1`

## Result

Nine factual regimes plus a reversal are admitted into ten opaque residual
slots. The shared transition model is trained once and frozen. Each lifetime
consumes `32` unique transition rows through one-pass random-feature
sufficient statistics, with complete-prefix held-out retention after every
promotion.

| seed | max prefix MSE | clean reliability | corrupt/OOD reliability | final capacity |
| ---: | ---: | ---: | ---: | ---: |
| 101 | 0.004544 | 0.9167 | 0.2500 | 10 |
| 102 | 0.014426 | 0.9583 | 0.2500 | 10 |

After four slots, an explicit copy-on-write growth transaction expands bank
capacity from `4` to `8` while preserving the content digest. Later admissions
grow capacity to `10`. A rejected growth proposal is a no-op. Opaque route
round-trips recover slots `0..9` after `45` existing-slot comparisons across
the ten novel bundles.

The learned external reliability statistics allow clean known reads and reject
both same-state corrupted evidence and a state outside the training range.
Those read probes do not mutate either the residual bank or the reliability
statistics; both router and verifier state persist exactly. Shuffled reversal
evidence is rejected.

Float16 compression passes held-out verification and reduces residual-bank
storage from `179,360` to `89,720` bytes. Int4 is rejected by the same
retention probe. The residual and reliability paths use zero replay; matched
fresh controls use `3,600` optimizer updates and replay `115,200` examples.

This promotes bounded capacity-scaled factual memory with replay-free learned
reliability. It does not establish general continual learning, arbitrary new
computation, or unrestricted memory growth.

Reports: `seed-101.json`, `seed-102.json`.

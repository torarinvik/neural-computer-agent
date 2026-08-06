# Order-visible no-replay generated composition curriculum (2026-08-06)

Status: rejected; no weights or artifact promoted.

This reran the short six-phase curriculum after correcting the benchmark so
the order of the two generic primitive cues is learner-visible. The routed
external stack still trained each phase only on fresh examples for its newly
admitted composition and replayed zero old generated-composition examples.

Final retention was `0.3594`, `0.3281`, `0.3594`, `0.3125`, `0.6719`, and
`0.7344` for composition IDs `0` through `5`; no phase reached a stable `0.75`
prefix. The parent remained stable and its core digest was unchanged.

The benchmark ambiguity was real but was not the primary failure at this
budget. The next experiment should isolate each acquired composition as its
own append-only external artifact and learn an opaque router over those
artifacts, preserving old artifacts while acquiring only the new one.

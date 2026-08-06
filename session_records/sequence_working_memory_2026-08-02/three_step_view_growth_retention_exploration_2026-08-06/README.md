# Retention-aware three-step growth exploration (2026-08-06)

This exploration extends the two-seed promoted two-step boundary with a third
online executable view, `adjacent_xor`, and applies retention checks to the
three sequential replacements plus float16, int8, and packed-int4 storage
representations.

Both seeds completed the full audit: seven opaque views in one physical row,
three protected intermediate replacements, frozen controller/old route,
zero replay, and retention-safe float16/int8/int4 transactions. Seed `69316`
achieved a three-step route accuracy of `1.0000`; seed `69317` achieved
`0.9980`. Every declared gate passed for both seeds.

The initial seed-`69317` rejection is retained as
`rejected_seed69317_pre_stable_prefix_fix.json`. It exposed an accounting bug,
not a representation limit: retained floors were recorded as the raw minimum
probe outcome while the gate evaluated the stable cumulative-prefix minimum.
For `adjacent_xor`, the raw minimum was `0.6875` but the stable floor was
`0.7070`, above the `0.70` threshold. All promoted retention scores now use
the stable-prefix definition consistently, and a paired full-precision source
control is recorded in each report.

The promoted reports are `report_seed69316.json` and `report_seed69317.json`.

# Selective query-side prior for append growth (2026-08-07)

This promoted experiment tests a selective copy-on-write prior. New
extensions copy only the frozen base query projections and router query
encoders; candidate-key projections and matching state remain fresh. The
base remains frozen and extensions stay disabled until fresh verifier
evidence.

At 64 calibration updates per stage, the mixed `[1, 2]` audit passes both
seeds at `1.0000` post-failure unseen routing with known retention,
stage-local permutation, reload, frozen-core, reward-shuffled, and zero-
replay gates passing. The matched fresh control passes seed `69316` but falls
to `0.6667` on seed `69317`; fresh initialization required 128 updates per
stage for the two-seed promotion.

This promotes selective query-side transfer and a 50% reduction in the
two-seed calibration budget. Full accounting is in `sample_efficiency_ledger.json`;
report and control checksums are in `SHA256SUMS`.

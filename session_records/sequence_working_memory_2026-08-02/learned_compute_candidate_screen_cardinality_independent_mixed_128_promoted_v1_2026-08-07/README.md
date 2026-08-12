# Cardinality-independent append calibration at reduced cost (2026-08-07)

This promoted follow-up repeats the mixed `[1, 2]` append audit with 128
fresh calibration updates per stage instead of 256. The singleton stage uses
attempted-outcome calibration; the two-candidate stage uses pairwise ranking;
the base and earlier stage remain frozen.

Across seeds `69316` and `69317`, pre-activation unseen routing is `0.0000`
and post-failure routing is `1.0000`. Known routing, base/stage-local
permutation, exact reload, frozen-core, reward-shuffled null, and zero-replay
controls all pass. This halves the append calibration optimizer updates from
the previous promoted mixed audit while preserving the same behavior gates.

This promotes a sample-efficiency gain for bounded external growth. It does
not establish arbitrary new computation, open-ended memory compression, or
general continual learning. Full accounting is in `sample_efficiency_ledger.json`;
report checksums are in `SHA256SUMS`.

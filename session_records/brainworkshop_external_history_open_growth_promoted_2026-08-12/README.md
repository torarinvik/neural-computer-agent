# Variable external-history open growth (2026-08-12)

This promotion closes the next temporal-computation bottleneck in the
canonical Brain Workshop path. The external compute ABI now accepts a
variable-length, explicitly masked history of learned event tensors, relative
ages, and the current event. The history reader consumes records in causal
order (oldest to newest, then current) while preserving opaque relative
addressing and independent external-file replacement.

The controller and event encoder were frozen after construction. Each
external file was trained from fresh verifier lifetimes and scalar action
outcomes only. No replayed examples entered training, routing, or retention
checks.

Seeds 17 and 18 both passed the complete two-file promotion audit:

- `symbol_parity` and held-out `triplet_parity` were admitted and mastered;
- every direct and protected-prefix probe was `1.0000`;
- same-cue route reversal demoted the old file and preferred the replacement;
- old-file retention, exact route reload, unknown-context near-chance, and
  reward-shuffled null controls passed;
- controller/event-encoder digests and admitted-file digests stayed unchanged;
- replayed examples were zero.

Accounting per seed was 179,712 unique verifier bits, 15,072 logical
lifetimes, 384 optimizer updates, and zero replayed examples. The run remains
a bounded two-file external-computation promotion, not arbitrary program
induction, unrestricted memory growth, or general continual learning.

Raw reports were generated under `/tmp/history-growth-2-promote-seed17.json`
and `/tmp/history-growth-2-promote-seed18.json` during the audit; only compact
summaries are curated here.

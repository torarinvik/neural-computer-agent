# Bounded active reads over unbounded external history (2026-08-12)

This promotion separates external-memory growth from per-step retrieval cost.
Each file receives an append-only external history that keeps the full event
lifetime, while the calibrated Brain Workshop files request four records: the
three preceding records plus the current event. The current event is included
explicitly before execution and the record is appended only afterward. This
preserves the successful n-back causal boundary without making the storage
capacity equal to the active computation window.

The controller and event encoder were frozen. Files were trained from fresh
verifier lifetimes and scalar action outcomes only. No replayed examples were
used in training, routing, retention, reversal, or reload checks.

Seeds 17 and 18 both passed the complete five-file audit:

- `symbol_parity`, `triplet_parity`, `parity2`, `switch_binary`, and `nback2`
  were admitted;
- every direct probe and protected-prefix retention probe passed the mastery
  gate; `nback2` was `1.0000` on all four direct lifetimes for both seeds;
- every route selected and mastered the correct file;
- same-context reversal demoted the stale route, retained the old file, and
  preferred the replacement;
- route reload was exact; unknown-context accuracy stayed near chance;
- reward-shuffled feedback rejected mastery;
- controller, frontend, and admitted-file digests stayed unchanged after
  routing; replayed examples were zero.

This promotes a scalable external-memory/query boundary and five-file
outcome-only growth. It does not establish unrestricted learned temporal
dependency, learned compression, arbitrary program induction, or general
continual learning.

Raw reports were generated under `/tmp/history-growth-5-bounded-seed17.json`
and `/tmp/history-growth-5-bounded-seed18.json`; only compact summaries are
curated here.

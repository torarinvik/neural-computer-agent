# Partial and contradictory factored read routing

This archive records the five-seed continuation of the factored memory
lifecycle test with a read-only partial-evidence boundary. After four
nonlinear regimes were promoted, each seed routed five-row subsets from known
regimes correctly. A mixed bundle containing evidence from two different
regimes was rejected as `ambiguous`, and an empty bundle returned an explicit
`ambiguous` no-op. The router digest was unchanged by all three reads.

The new path requires a configurable fraction of agreeing rows and a separate
contradiction floor. It never stages a candidate or writes memory, so missing
evidence cannot silently become a new version and contradictory evidence cannot
be averaged into an existing version. Full-bundle admission remains the
separate verifier-gated write path.

All five seeds also retained the earlier lifecycle gates: capacity growth,
compression round-trip, stable-ID middle eviction, new-slot admission, and
persistence. This promotes a bounded read-safety mechanism, not automatic
open-world version formation or general continual learning.

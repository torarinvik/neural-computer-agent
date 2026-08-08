# Two-generation page router with source-only normalization

Promoted replicated audit of the 66-candidate, 21-page learned opaque page
router. The signature normalizer is fit only on the original 30 source keys;
future generation keys are never used to define the frozen representation.
Each append generation has an independently trained token-preserving router,
and the verifier-gated cascade is source → generation 1 → generation 2.

Both seeds (69316 and 69317) pass every strict gate at the matched 4,096
updates-per-generation-router budget: candidate and page mastery, per-target
and per-page floors, permutation, generation-local reward-shuffled nulls,
source immutability, controller invariance, no unresolved rows, exact reload,
and zero replayed examples.

Per seed: 19,040 optimizer updates, 1,986,816 unique verifier bits, and
1,986,432 unique logical lifetimes. The earlier 3,072-update source-only
boundary was not promoted because seed 69317 missed one target; this archive
records the matched budget that closes that specific boundary.

This is bounded repeated external growth with a source-only representation
contract. It does not establish unrestricted memory growth, compression,
arbitrary new computation, or general continual learning.

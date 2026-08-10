# Promoted verifier-gated masked identity-memory growth

This four-seed audit starts external identity memory with one prototype per
slot, rejects an unsafe capacity-growth transaction, then accepts growth from
one to three prototypes per slot only after a copy-on-write retention probe.
Two distinct masked observations with different partial-evidence masks are
subsequently appended for one slot; the original full identity remains intact.

All runtime updates use opaque anchor proposals. The controller, transition
model, verifier statistics, and alignment adapters remain frozen. Persistence
is exact and no old examples are replayed.

Results across seeds `85201`–`85204`:

- affine mastery: `1.0` for every seed;
- nonlinear mastery: `0.9917`–`1.0`;
- prototype counts: slot `0` grows to `3`, slot `1` remains at `1`;
- rejected growth preserves the source digest;
- accepted growth retains both full routes and both partial routes;
- final memory round-trips exactly; replay count is `0`.

The mask-overlap compatibility gate prevented false merges when the second
mask shared only half of the union of observed dimensions. The result is
bounded verifier-gated external-memory growth, not autonomous retention policy,
unbounded memory, semantic open-world identity, or general continual learning.

# Stable value path with persistent reload — seed 69415

Status: promoted narrow persistence replication.

With the phase-1 extension required by parent stability, the same external
writer and persistent backend achieved reload recall `0.996`, rejected checksum
corruption, and recovered at `1.000`. Retention remained target-first `0.986`,
target-last `0.991`, with zero replayed examples.

# Promoted bounded masked-memory maintenance stream

This four-seed audit exercises repeated external-memory maintenance over a
28-step online stream. One identity slot grows from two to five prototypes,
learns four differently masked patterns, sees forward and reverse order three
times, rejects and accepts a replacement, rejects and accepts a consolidation,
then re-admits a reversed pattern after compression.

The opaque capacity planner receives a side-effect-free candidate view at a
maintenance point. It is advisory and untrained; all state changes still pass
the explicit verifier-gated growth, replacement, or consolidation APIs.

Results across seeds `85401`–`85404`:

- forward/reverse stream routes: all passed;
- affine mastery: `1.0` for every seed;
- nonlinear mastery: `0.9833`–`1.0`;
- rejected maintenance transactions preserved source digests;
- accepted replacement and consolidation retained their required routes;
- final persistence was exact; replay-buffer reuse was `0`.

This promotes bounded online maintenance under changed masks and reversal. It
does not establish a trained capacity policy, autonomous retention/compression,
unbounded memory, semantic open-world identity, or general continual learning.

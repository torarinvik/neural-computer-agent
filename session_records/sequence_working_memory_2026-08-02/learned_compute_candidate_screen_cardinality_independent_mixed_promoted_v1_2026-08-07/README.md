# Cardinality-independent append calibration (2026-08-07)

This promoted experiment combines pairwise and unary external-screen
learning. Seven mastered candidates remain in a frozen base; three
outcome-unseen candidates are split across a singleton first extension and a
two-candidate second extension. The singleton is trained with fresh scalar
verifier outcomes for attempted candidates, including positive and negative
attempts across fresh contexts. The two-candidate stage retains pairwise
ranking. Later activation still requires cumulative failure of the base and
earlier stage.

Across seeds `69316` and `69317`, pre-activation unseen routing is `0.0000`
and post-failure routing is `1.0000`; known routing, base/stage-local
permutation, exact reload, frozen-core, reward-shuffled null, and zero-replay
controls all pass.

This promotes cardinality-independent bounded append calibration. It does not
establish unbounded memory growth, arbitrary new computation, open-ended
compression, or general continual learning. Full accounting is in
`sample_efficiency_ledger.json`; report checksums are in `SHA256SUMS`.

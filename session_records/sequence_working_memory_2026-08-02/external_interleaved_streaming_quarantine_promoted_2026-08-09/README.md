# Interleaved streaming candidates with bounded quarantine — promoted

Across seeds `1901` and `1902`, two novel affine dynamics streams arrived in
alternating four-row windows before either candidate was promoted. The router
isolated both candidate sufficient-statistics models, selected the affine
family through held-out factual verification, retained no raw rows inside
candidate state, and refused a third stream at capacity.

An intentionally over-conservative margin quarantined one clear-but-not-yet
trusted four-row bundle. A later margin check assigned it to the correct
candidate, and the candidate consumed the deferred bundle exactly once. The
quarantine was persisted and restored before resolution; promotion was refused
while it remained unresolved. All controller, retention, persistence, causal
shuffle, capacity, and zero-replay gates passed on both seeds.

This promotes bounded transactional ambiguity handling for streaming factual
candidates. It does not establish arbitrary nonlinear one-pass learning,
unrestricted growth, or general continual learning.

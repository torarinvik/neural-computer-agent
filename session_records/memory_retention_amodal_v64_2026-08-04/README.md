# v64 fixed warmup token schedule

v64 keeps fixed tokens through parent acquisition and the 64-step retention
warmup before enabling token randomization. Seed 18 preserves narrow retention
and unseen-token recall, but its fresh transfer population still fails. The
warmup boundary is not promoted as a transfer solution.

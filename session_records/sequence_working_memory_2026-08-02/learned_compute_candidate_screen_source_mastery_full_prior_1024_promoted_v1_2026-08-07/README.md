# Source mastery plus full prior promoted (2026-08-07)

This is the strict bank-20/five-stage boundary after doubling only the source
screen budget from 512 to 1024 updates. Full append priors copy the mastered
factorized screen into each isolated extension; the parent controller remains
frozen during screen and extension learning.

The full-prior configuration passes both seeds under strict per-candidate
promotion: known and unseen routing are `1.0000/1.0000`, every audited target
clears the `0.75` floor, and reload, permutation, frozen-core,
reward-shuffled, and zero-replay controls pass. This promotes a bounded
source-mastery-plus-prior result, not general continual learning.

The matched fresh-extension control passes seed `69316` but fails seed `69317`
with one unseen target at `0.0` and aggregate unseen routing `0.8958`. The full
prior therefore remains justified as a robustness mechanism at this boundary,
although the result does not claim a universal sample-efficiency multiplier.

Accounting per run is 1,216 optimizer updates, 250,368 unique verifier bits,
249,984 unique logical lifetimes, and zero replayed examples. Candidate-key
separation remains weak on the harder seed (effective rank `3.59`), so source
mastery is repaired by budget and prior transfer but representation quality is
still an open bottleneck.

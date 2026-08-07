# Spatial-binding frontend control rejected (2026-08-07)

This control replaces the historical average-pooled visual frontend with a
learned frontend that preserves the ordered coarse feature map before emitting
the same 32-wide event tensor. The frozen controller, factorized screen,
twenty-candidate bank, five two-candidate append stages, and fresh budgets are
otherwise unchanged.

The representation diagnostic improves, but the capability result does not.
Nearest-neighbor key cosine falls to `0.9778/0.9860` at the worst rows and
effective rank rises to `7.01/6.16`, compared with `0.9982` and `3.59` for the
weaker average-pooled baseline seed. However, unseen routing is only
`0.8958/0.8021`, with a strict per-target hole in both seeds, versus the
factorized baseline's `1.0000/0.8958`. The frontend is therefore rejected as a
promoted acquisition path.

The result retains the spatial-binding blueprint as a diagnostic lead: better
separation is not sufficient unless query/key alignment and downstream
generalization improve as well. No replay is used, and both reports include
the frozen-core, reload, permutation, reward-shuffled, strict per-target, and
verifier-bit/lifetime controls.

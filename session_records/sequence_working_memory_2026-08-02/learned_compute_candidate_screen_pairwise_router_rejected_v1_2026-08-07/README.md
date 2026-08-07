# Pairwise screen router rejected (2026-08-07)

This control replaces the factorized query/key dot-product router with a
shared permutation-equivariant pairwise router while holding the frozen
controller, bank size, five two-candidate stages, and 32-update budget fixed.

The result is not promoted. Unseen routing is `0.9063/0.7083` across seeds;
the second seed fails the strict acquisition and permutation gates. The
factorized baseline at the same rung is `1.0000/0.8958`. The more expressive
pairwise scorer therefore adds complexity without improving the current
boundary. The branch is discarded from the canonical API; this archive keeps
the negative control reproducible.

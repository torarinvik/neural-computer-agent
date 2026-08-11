# Ordered credit-assignment contrast — 2026-08-11

This is a matched diagnostic of the next bottleneck identified after the
shared-composition and operator-algebra failures: binding an ordered sequence
of opaque external fragments to the final intention. The new trainer arm
reuses each rendered example with a cyclically shifted route and applies a
trainer-only inverted counterfactual loss. The parent, acquired fragment bank,
and all verifier metadata remain outside the deployed learner.

## Transport correction

The curriculum path exposed two real accounting/transport defects before the
learning comparison was trusted:

- mixed-depth programs were grouped by executable length before batching;
- composition IDs were made contiguous per target, so reported target scores
  are actually target-specific rather than averages over alternating rows.

These are retained as infrastructure fixes. The prior mixed-row reports are
not used as per-target evidence.

## Matched result

Both arms used seed `69316`, 16 parent updates, 64 updates for each of four
primitive fragments, 64 composition updates, batch size 16, span 3, audit
count 32, and the segment combiner. The counterfactual arm used
`--order-contrast-weight 0.5` and applied the same extra loss to the shuffled
and fresh controls.

| metric | baseline | order contrast |
| --- | --- | --- |
| shared training accuracy | `0.6458 / 0.6875 / 0.7813` | `0.6979 / 0.6875 / 0.7917` |
| held-out order accuracy | `0.5313 / 0.6250 / 0.6354` | `0.5104 / 0.5729 / 0.4792` |
| mean held-out accuracy | `0.5972` | `0.5208` |
| wrong-order accuracy | `0.5417 / 0.6667 / 0.7188` | `0.3542 / 0.5521 / 0.3125` |
| stable shared/fresh bits | none / none | none / none |
| unique verifier bits | `87,456` | `87,456` |
| optimizer updates | `464` | `464` |
| paired counterfactual rollouts | `0` | `576` |
| wall time | `73.80 s` | `116.25 s` |

The contrast improved the wrong-order margin but reduced held-out order
generalization by 7.64 percentage points and did not produce a stable prefix.
The capability promotion is therefore rejected. Frozen-parent, frozen-bank,
no-bypass, missing-evidence, reward-shuffled, persistence, and zero-replay
controls passed.

## Decision

Retain the length-grouped transport, contiguous target accounting, and
optional counterfactual-loss hook as diagnostic infrastructure. Do not promote
the order contrast or make it part of the default learner. The result says
that ordered credit assignment is real, but a route-level negative loss is too
coarse: it teaches rejection of some shifted routes without teaching the
shared learner a reusable ordered execution law.

The next high-ROI experiment should reuse trace computation and expose
operator-level intermediate verifier signals or a protected, step-indexed
external execution state. It must preserve fresh-order, wrong-order,
missing-evidence, memory-corruption, and stable-prefix controls.

Claim boundary: this is not arbitrary program induction, unrestricted memory
growth, compression, or general continual learning.

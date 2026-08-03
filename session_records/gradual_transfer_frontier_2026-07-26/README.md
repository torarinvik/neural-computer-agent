# Gradual transfer frontier — 2026-07-26

## What passed

Correct accumulated physical history causally accelerates a related held-out
utility relation.

Across seeds 7032 and 7034:

- intact history beat shuffled visible history on normalized verified-reward
  AUC in both replicas;
- intact history beat shuffled history on target-selection AUC in both;
- intact history reached a +10-point target advantage in the first 288
  candidate verifier bits in both replicas;
- shuffled history never reached that threshold; and
- binary and four-rule retention remained accepted.

The verifier truth and physical banks were kept intact during corruption. Only
the controller-visible access/success/failure features were shuffled. An
earlier malformed audit that changed both the history and target was discarded
and overwritten.

Canonical report:
`experiments/archive/unified_cognitive_controller/reports/gradual_transfer_audit_seeds7032_7034.json`

## What did not pass

Reusing the single global two-weight utility residual did not produce robust
cross-relation compounding:

- on the standard curriculum, warm beat cold on seed 7032 but tied seed 7034;
- a `0.2 -> 0.4` reliability curriculum was too easy at the source and warm
  was identical to cold;
- a `0.3 -> 0.4` curriculum learned the source, but warm transfer was worse
  than resetting the residual before the target.

This is negative transfer, not insufficient target duration. The source update
compresses one utility condition into a single global residual and overwrites
rather than stores a reusable strategy.

## Next highest-ROI architecture atom

Keep one controller, but move rapid utility adaptation into context-indexed
fast state:

1. controller encodes the current unlabeled physical-history distribution;
2. RAM holds a small bank of learned utility-strategy latents;
3. content addressing retrieves or interpolates a strategy;
4. verified reward updates the selected latent, not one global residual;
5. useful latents consolidate to disk;
6. controller weights remain shared and slow.

First test with two nearby reliability contexts. Warm retrieval must beat:

- a global-residual learner;
- an empty strategy bank;
- shuffled strategy keys;
- reset fast state; and
- a frozen controller.

Pass only if it reaches a fixed target advantage in fewer verifier bits on two
seeds while binary/four-rule retention remains accepted.

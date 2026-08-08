# Pass bars, stated before the runs report

Written 2026-08-08 while all four runs are still training. The point is
that the bar cannot move to fit whatever comes back.

## A. Ignorance diagnostic (`ign2diag-{69316,69317}.json`)

Re-runs of the already-reported `--ignorance 2.0 --ignorance-every 1`
config, deterministic, so training reproduces bit-for-bit and only the
report gains fields. This is NOT a new attempt to pass the gate — the
gate already failed (decoy choiceA 0.969/0.906). It is a mechanism test.

**Hypothesis:** the ignorance objective flattened the decoy policy's
confidence without reordering its logits, so greedy argmax still
recovers the bank-free default.

Confirmed if, for choiceA under decoy:
- `decoy_entropy` is near ln(4) = 1.386 (the policy is close to uniform), AND
- `decoy_max_prob` is near 0.25 but above it (a residual tilt survives), AND
- `decoy` (greedy) stays high while `decoy_sampled` is materially lower.

That combination means the *stochastic* policy is genuinely ignorant and
only the deterministic readout leaks the default. It would make the
finding "output-space penalties cannot reorder logits", and would also
oblige us to re-check whether earlier decoy gates were run greedily.

Refuted if entropy is far from uniform — then ignorance never took hold
at all and the story is simply "not enough pressure", which the 4x run
already argues against.

## B. Symmetric plant (`sym-{69316,69317}.json`)

New mechanism: both contexts rolled out every update, one step on the
sum, so the plant only ever receives the mixture gradient (Galashov-style
information asymmetry). Matched to A's config exactly.

**Promotion bar — all of it, on BOTH seeds:**
- both twins mastered (train >= ~0.9 against a 1.000 solo ceiling)
- cross-feed inverting to ~0.000 in both directions
- **decoy collapsing to chance for BOTH twins** — this is the gate that
  has failed every previous attempt, and choiceA is the one that fails

Anything less is not a pass. In particular:
- both twins mastered + choiceA decoy still high = same failure as F48,
  recorded as such, mechanism rejected.
- decoy collapses but a twin fails to master = the mixture gradient
  cancelled too much to learn from. That is an interesting and reportable
  negative (it would mean the plant cannot be made context-neutral and
  still be trainable on contradictory twins), not a pass.

**Prediction, on the record:** the risk is the second failure mode. F50
measured these two gradients as genuinely conflicting, so summing them
cancels much of the signal. The mechanism only works if what survives
the cancellation — "read the fragment and do what it says" — is itself
learnable, since that direction IS common to both contexts. If mastery
collapses, that is the finding.

## Reporting rule for both

Gates first, bar stated above, matched configurations only. No presenting
one seed's config alongside another's.

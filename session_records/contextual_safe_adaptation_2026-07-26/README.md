# Contextual safe-adaptation estimator — pre-registration

## Goal

Reduce the replicated safe-promotion delay below the current 480–720 verifier
bits without weakening the positive lower-95%-confidence promotion rule or
causing any mastered-policy promotion.

## One-axis change

The existing paired inverse-propensity estimator subtracts one global mean
outcome. The candidate instead fits a tiny ridge-linear context baseline from
the four generic memory statistics. Two-fold cross-fitting ensures each
record's baseline prediction is produced by a model that did not train on that
record's outcome.

The baseline is independent of the logged action and therefore cannot encode
the unattempted result. Subtracting a context-only baseline leaves the expected
incumbent-versus-challenger policy difference unchanged under the randomized
logger, while potentially removing context-dependent outcome variance.

## Gate

Seed 7961 uses the unchanged 720-bit capacity-six experiment. It passes only
if:

1. the mastered incumbent receives zero promotions and retains utility;
2. the gap challenger is promoted with a positive lower 95% bound;
3. first promotion occurs before 480 bits, strictly improving on the best
   replicated global-baseline result;
4. retention and persistence checks pass.

Only a full pass permits unchanged seed-7962 replication. Otherwise the
contextual estimator is closed without a longer run.

## Result

Seed 7961 remained safe:

- mastered incumbent: zero promotions and exactly unchanged utility;
- gap learner: promoted with lower confidence bound `+0.0203`, improving
  audited utility by `6.37` points.

However, promotion occurred only at 720 verifier bits. This does not beat the
480-bit replicated global-centered baseline, so the sample-efficiency gate
failed. No replication or longer run is allowed.

The cross-fitted contextual estimator is closed. The global centered estimator
remains the best verified safe-adaptation mechanism. The higher-ROI frontier is
now integrating incumbent/challenger promotion with persistent skill storage,
so safely promoted knowledge survives future task sequences.

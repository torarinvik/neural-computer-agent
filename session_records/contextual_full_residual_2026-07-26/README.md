# Context-retrieved full-feature residual

Date: 2026-07-26

## Why this experiment exists

The redundancy atom was causally learnable only after resetting the complete
replacement policy.  That reset rose from 71.88% to 86.46%, then fell to
76.04% when row novelty was shuffled.  However, globally resetting the old
policy violates retention.

This experiment tests the smallest architecture that can express both facts:

- preserve the old replacement policy unchanged;
- learn a new local correction that can override any of its generic feature
  contributions in a sufficiently similar visible context.

## Architecture

The shared controller remains frozen.  A bounded external strategy memory
stores:

- a 12-dimensional context key: mean and standard deviation of six generic
  visible row statistics;
- an eight-dimensional full-feature residual.

For a memory decision:

```text
scores = old_replacement_policy(features)
       + features · retrieved_residual
```

The residual is initialized to exact zero.  Retrieval activates only when
cosine similarity is at least `0.982`; otherwise the returned residual is
exactly zero.  A read-only preflight on disjoint generated contexts found:

- new↔new similarity: 0.9871–0.9969;
- new↔old similarity: 0.9513–0.9786.

The threshold was fixed before training.  It uses no task identifier or
semantic label.

## Sub-minute protocol

- unseen stream seed: 7311;
- deterministic capacity-three redundancy atom;
- eight black-box updates;
- 16 banks per update;
- three verifier-scored candidates: positive, center, negative;
- up to 16 reward-free direction proposals, selected only for action
  disagreement;
- bounded strategy capacity four;
- save/reload after every update.

Arms:

1. context-retrieved intact residual;
2. identical reward-alignment-shuffled residual;
3. globally active residual, to expose retention cost;
4. unchanged old policy.

All arms share generated experiences, direction proposals, verifier budget,
and frozen controller weights.

## Gates

Promote only if:

- the intact contextual arm reaches a stable 25% oracle-gap threshold;
- it finishes at least two verified-reward points above the unchanged policy;
- shuffling novelty removes at least four points;
- reward-aligned training beats reward-shuffled training;
- held-out new contexts activate retrieval;
- old equal, reliability-dominant, and old-return contexts reject retrieval
  and remain bit-identical to the old policy;
- every strategy-memory save/reload is exact;
- binary mapping and four-rule retention pass.

Target-row labels remain diagnostic only.  Training receives only attempted
candidate outcomes.  Failure stays at the sub-minute rung.

## Seed-7311 result

The first run finished in 22.4 seconds and did not pass the learning gate.
Nevertheless, the architecture plumbing worked:

- held-out new-context retrieval acceptance: 100%;
- old equal, reliability-dominant, and return contexts: all rejected;
- old-context replacement scores: bit-identical;
- every strategy save/reload: exact;
- every round found action-divergent candidates;
- aligned reward AUC exceeded reward-shuffled AUC.

Verified reward briefly rose from 81.25% to 83.85% at update two, then
oscillated back to 82.29% by update eight.  Novelty shuffling cost only 2.08
points, below the four-point gate.  The global and contextual arms matched on
the new context, as expected with one active residual slot; their difference
appeared on old contexts, where only the contextual arm was an exact no-op.

This is an active-but-coarse optimizer signature, not evidence for a longer
run.  Pre-register one sub-minute scale correction on unseen seed 7312:

- keep perturbation at 8.0 so candidate actions remain distinguishable;
- reduce the committed step from 4.0 to 1.0;
- keep every other budget, threshold, arm, and gate unchanged;
- log candidate rewards, winner, and residual norm at every update.

If the smaller step does not pass, stop tuning this SPSA configuration.

Seed 7312 with the smaller step also failed: contextual reward declined from
80.21% to 78.65%, aligned AUC did not beat the shuffled control, and novelty
shuffling had no effect.  Full eight-dimensional SPSA is rejected at this
sub-minute budget.

## Pre-registered low-dimensional arbitration adapter

The successful complete reset reveals a much smaller sufficient solution:
suppress the inherited policy and use the new generic statistic.  Test a
two-dimensional context-retrieved adapter on unseen seed 7313:

```text
scores = exp(clamp(log_old_scale, -6, 3)) * old_scores
       + novelty_weight * centered_row_novelty
```

The zero vector is handled as an exact bypass, so old contexts remain
bit-identical.  This adapter does not encode a correct action, target row, or
semantic task identifier.  It learns only how much to trust an existing policy
versus one newly exposed generic memory statistic.

Use perturbation 3.0 and committed step 1.5 in the two-dimensional space.  Keep
the same eight updates, contexts, verifier accounting, retrieval threshold,
controls, and gates.  This is a new mechanism justified by the dimensionality
localization, not another scale tune.

Seed 7313 produced a strong partial result:

- verified reward: 82.29% → 84.38%;
- novelty-shuffled reward: 79.17% (5.21-point causal loss);
- reward-aligned AUC beat the shuffled arm;
- every old context remained bit-identical.

It missed only the stable oracle-gap threshold.  The per-update trace exposed
a correctness bug: ordinary `argmax` selected the positive candidate whenever
positive, center, and negative rewards tied.  Updates 7 and 8 therefore moved
the residual without any verified improvement.

Fix candidate selection before another run: the unchanged center wins unless
another candidate exceeds it by more than `1e-6`.  On unseen seed 7314, repeat
the seed-7313 configuration unchanged with this verified-improvement rule.
This is an optimizer correctness repair, not a hyperparameter fork.

Seed 7314 did not replicate learning.  Its trace localized a routing collision:
one new-context batch missed the threshold and created a second slot; later
batches alternated between the learned slot and a zero slot.

A 50-new/50-old read-only key audit showed the original normalized mean/std key
had overlapping tails:

- new↔new minimum: 0.9722;
- new↔old maximum: 0.9844.

Add one generic statistic for every keyed feature: whether that feature is
active (mean absolute magnitude above `1e-6`).  This does not identify a task
or utility; it records the presence of generic inputs.  The resulting
18-dimensional key gave:

- new↔new minimum: 0.9936;
- new↔old maximum: 0.9294.

Keep the already registered 0.982 threshold.  Repeat the two-dimensional
adapter unchanged on unseen seed 7315.  Promotion still requires every
behavioral, causal, persistence, and exact-retention gate.

Seed 7315 confirmed that the activity key solved routing but did not recover
SPSA learning.  Every new context retrieved, all old-context similarities fell
near 0.926 and were rejected, yet contextual reward stayed at 83.85%.
Two-dimensional SPSA is rejected rather than promoted.

## Pre-registered per-bank policy-gradient rung

The black-box race compresses 16 bank outcomes into three candidate means.
Test a more sample-efficient use of the same allowed signal on unseen seed
7316: REINFORCE over the two adapter parameters.

- sample one attempted replacement per bank;
- receive only that bank's later scalar verified outcome;
- subtract the within-batch mean as a generic baseline;
- temperature 2.0, learning rate 1.0, entropy bonus 0.01;
- eight updates and 16 banks remain unchanged.

This consumes one-third the candidate verifier bits of SPSA.  The
reward-shuffled arm permutes outcomes across banks before the gradient, testing
action–outcome alignment.  No target row or unattempted-action reward is
provided.  All routing, persistence, novelty-shuffle, and retention gates stay
unchanged.

Seed 7316 used only 384 candidate verifier bits and produced nonzero gradients
on every update, but the adapter norm reached only 0.058 after eight updates.
No held-out action changed.  This localizes failure to gradient scale.

Run one final pre-registered check on unseen seed 7317: normalize each
two-dimensional policy gradient to unit norm and commit a 0.5-norm update.
Direction, sampled outcomes, routing, and all budgets remain unchanged.  This
is generic gradient normalization, not task information.  If it does not pass
the full gate, reject this adapter/training combination.

## Final result

Seed 7317 again preserved every routing, persistence, and inherited cognitive
gate, but failed the behavioral gate:

- contextual reward: 84.38% → 84.90%;
- reward-shuffled final: 83.85%;
- novelty-shuffled contextual: 84.38%;
- no stable oracle-gap threshold.

The adapter/training combination is rejected at the sub-minute rung.  No
three-minute run is justified.

## What is now established

The context mechanism itself is successful:

- feature-activity keys separate new and old contexts with a wide margin;
- new contexts retrieve learned external state;
- old contexts reject it and remain bit-identical;
- disk save/reload is exact;
- routing adds no catastrophic forgetting.

What remains unsolved is low-variance credit assignment from attempted scalar
outcomes.  SPSA was unstable; ordinary REINFORCE produced gradients too small
to change behavior; normalized REINFORCE changed behavior slightly but did not
learn a causal novelty policy.

The next experiment should not extend these runs.  Train a tiny
action-conditioned success critic passively from only:

- visible generic memory statistics;
- the action actually attempted;
- its exact logging probability;
- its later scalar verifier outcome.

The critic must first pass held-out calibration, reward-shuffle, and
missing-evidence gates while having no influence on actions.  Only then may it
guide the context adapter.  This uses the same sparse information more
efficiently without inventing labels or rewards for unattempted actions.

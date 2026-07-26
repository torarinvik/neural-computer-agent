# Redundancy-utility forward-transfer microexperiment

Date: 2026-07-26

## Question

Does the state selected by the balanced maximin population race reduce the
verified experience needed to learn one genuinely later memory-utility
primitive?

The later primitive is **row novelty**.  The bounded memory already exposes
age, access frequency, write strength, candidate similarity, and empirical
reliability.  This experiment adds one generic statistic: how dissimilar a
stored row is from the other stored rows.  Low-novelty rows are redundant and
therefore cheaper to replace.

No task name, utility weight, target row, correct action, or unattempted-action
label is visible to the learner.  The learner sees only its ordinary latent
memory statistics and the scalar verified success produced after its chosen
replacement.

## One-axis curriculum

The first diagnostic uses:

```text
recency     0.30
frequency   0.30
reliability 0.30
novelty     0.10
```

Only novelty is new.  If the sub-minute diagnostic produces a causal learning
signal, later stages may raise novelty gradually.  They must not change another
difficulty axis in the same promotion.

## Matched arms

Every arm receives the same generated memories, future queries, verifier
outcomes, perturbation directions, and update budget.

1. `selected_experience`: replacement weights from the exact stream-7085,
   clone-7211 balanced-maximin winner, expanded by a zero novelty coefficient.
2. `shared_parent`: the population's common seed-6810 parent, expanded with
   zero reliability and novelty coefficients.
3. `architecture_reset`: the selected architecture on the identical frozen
   controller backbone, with all three residual utility coefficients reset to
   zero.
4. `fresh_matched`: a fresh matched replacement policy initialization.  It is
   a lower-bound control, not a retention candidate.
5. `selected_reward_shuffled`: the selected initialization with candidate
   verifier outcomes misaligned before each update.

This first rung isolates transferable **weights**.  It does not yet claim that
the winner's physical disk rows or strategy-memory entries transfer.  Those
remain a later rung if weight transfer is positive.

## Accounting

Record per arm:

- unique generated logical lifetimes;
- candidate verifier bits;
- optimizer-free black-box updates;
- replayed examples;
- wall time;
- held-out reward at every measured prefix;
- the first threshold that remains satisfied at all later prefixes;
- transfer ratio against the fresh policy;
- retention on the old utility and inherited cognitive primitives.

Thresholds are fractions of the held-out verified reward gap between the
strongest nonlearned control and the visible-statistics oracle.  Target-row
accuracy is diagnostic only and cannot select or train an arm.

## Sub-minute promotion gate

Promote to roughly three minutes only if all of the following hold:

- the visible-statistics oracle has a nontrivial verified-reward advantage;
- some intact arm produces a stable above-baseline learning signal;
- the selected-experience arm reaches at least one stable threshold using
  fewer verifier bits than every matched nonexperienced arm;
- its final old-utility performance remains within two reward points of its
  own prefix-zero value;
- shuffling novelty at evaluation damages the learned policy, or the
  reward-shuffled training control fails to reproduce its learning curve;
- binary mapping and four-rule retention remain accepted.

If the selected weights lose to the reset arm, retain the architecture and
reset the weights for this primitive.  If every arm remains flat, first reduce
the curriculum jump before changing architecture.

## Escalation

The ladder is fixed in advance:

1. sub-minute diagnostic;
2. approximately three minutes only after the gate above;
3. approximately ten minutes only after an unchanged unseen-seed replication;
4. only then integrate the third utility dimension into physical disk and
   latent strategy memory.

## Sub-minute result: weights-only rung

The first completed run used seed 7301, eight updates, 16 training banks, and
32 held-out banks.  It finished in 28.3 seconds.

The selected winner's global residual did **not** transfer:

- selected weights: 89.58% final verified reward;
- shared parent: 93.23%;
- zero-reset architecture: 91.67%;
- fresh matched replacement policy: 90.62%;
- visible-statistics oracle: 94.79%.

Only the shared parent crossed stable 25% and 50% oracle-gap thresholds, at
1,152 candidate verifier bits.  The three-minute promotion gate therefore
failed.

This is a bounded negative for weight-only compounding.  Inspection then found
that the selected checkpoint's deployed policy was not its global residual
alone: its successful physical run retrieved among four context-indexed
two-dimensional strategies.  The weights-only arm discarded that learned
state, so it cannot decide the full-state transfer question.

## Pre-registered full-state retrieval rung

Before running another seed, add one arm:

`selected_strategy_memory`

It uses the selected checkpoint's saved context encoder, prior scalar reward
signature, and bounded strategy bank to retrieve one old strategy from the new
context.  The retrieved two coefficients are preserved and the new novelty
coefficient is initialized to zero.  No utility label or target action enters
retrieval.

The global selected residual remains as a diagnostic arm.  All other arms,
threshold definitions, verifier accounting, and promotion gates stay
unchanged.  The old-utility retention comparison must use the same fixed
held-out old-utility data before and after adaptation; comparing unlike task
distributions is invalid.

## Full-state retrieval result

The unseen seed-7302 retrieval run also finished in 28.3 seconds.  The saved
strategy bank retrieved an old two-dimensional policy and improved over the
winner's global-residual-only arm, but it did not beat the common parent:

- selected strategy memory: 95.83% final verified reward;
- selected global weights: 94.79%;
- shared parent: 96.35%;
- zero-reset architecture: 95.83%;
- fresh matched policy: 94.27%.

The retrieved strategy reached the stable 25% gap threshold at 1,152 candidate
bits.  The parent was already above the 25% and 50% thresholds at prefix zero.
The compounding promotion gate therefore failed again.

Novelty shuffling did not damage the leading policies.  This means the 10%
novelty mixture was not a valid test of acquired redundancy use: old utility
features could still carry the result.

## Pre-registered one-axis correction

Keep the sub-minute budget, architecture, arms, perturbations, and gates fixed.
On unseen seed 7303, raise only novelty from 10% to 20% and divide the remaining
80% equally across recency, frequency, and reliability.  This is a curriculum
calibration, not a duration promotion.

If novelty remains noncausal, do not claim redundancy learning.  If novelty is
causal but the selected state still loses to its parent/reset controls, reject
retained population weights for this primitive and continue from the simpler
parent or reset policy.

The 20% run produced the opposite boundary: no intact arm finished above the
strongest simple control, while novelty shuffling began to damage several
policies.  This is consistent with a curriculum step that became load-bearing
but was too large for eight updates.

Before changing duration or mechanism, test the single midpoint on unseen seed
7304: novelty 15%, with the remaining 85% divided equally among the three old
utilities.  Everything else remains fixed.  This midpoint is the final
sub-minute curriculum calibration; if it does not show a clean signal, stop
the fork rather than tuning continuously on held-out results.

The midpoint again failed the promotion gate.  The parent alone produced a
small stable 25%-gap crossing; neither selected-state arm crossed, and novelty
shuffling did not reduce the parent's score.

A read-only distribution audit then found the task-construction problem:
across the 10%, 15%, and 20% mixtures, the redundancy-only action agreed with
the realized target only 19.5–25.0% of the time (six-way chance is 16.7%).
Thus the curriculum asked the learner to discover a weak fourth factor before
the fourth factor had ever been mastered alone.

## Pre-registered redundancy atom

Run one final sub-minute localization on unseen seed 7305:

- capacity three;
- novelty weight 100%;
- zero utility noise;
- identical eight updates and 16 banks;
- all matched arms and causal controls retained.

This is not a claim of useful mixed memory management.  It asks only whether
the controller can acquire the deterministic redundancy atom from scalar
outcomes.  Passing requires a stable verified-reward crossing and a damaging
novelty shuffle.  If the atom passes, the next curriculum may add old utility
one component at a time.  If it fails, stop and inspect the action/reward
interface rather than scaling.

The atom run caught an accounting defect: the threshold baseline included the
hand-written redundancy policy, which is the new primitive's oracle.  That
made the available oracle gap zero by construction.  The corrected baseline
contains random, fixed, skip, recency, frequency, and reliability controls;
redundancy remains an upper-bound diagnostic.

Even under the corrected interpretation, no learned arm beat the strongest
old-feature control in eight updates.  However, the fresh replacement policy
rose from 77.08% to 87.50%, and novelty shuffling reduced it to 75.00%.  That
is the first causal evidence that the new coefficient is learnable.  The
controller-preserving reset arm stayed flat because it reset only the linear
residual; it retained the old nonlinear five-feature replacement gate.

## Pre-registered complete-policy reset

On unseen seed 7306, repeat the redundancy atom unchanged and add one arm:

`replacement_policy_reset`

It preserves the sensory encoder, recurrent controller, abstract intentions,
memory read path, and eight-feature replacement architecture, but zeros both
the old nonlinear replacement gate and the linear residual.  Only this tiny
action-selection policy is reset.

This arm passes the atom's mechanistic gate only if it finishes above the
strongest non-redundancy control and loses at least two verified-reward points
when novelty is shuffled.  The selected-state forward-transfer gate remains
separate; a reset-policy success does not retroactively count as compounding
from selected weights.

## Complete-policy reset result

The unseen seed-7306 run passed the atom's mechanistic gate in 23.8 seconds:

- complete replacement-policy reset: 71.88% → 86.46%;
- strongest non-redundancy control: 85.42%;
- novelty-shuffled reset policy: 76.04%;
- redundancy oracle: 96.88%.

The 10.42-point causal loss under novelty shuffling is substantially larger
than the reset policy's 1.04-point margin over the strongest old control.  The
atom is therefore learnable from scalar verified outcomes, and its learned
policy genuinely uses the new statistic.

The selected global residual, selected strategy retrieval, shared parent, and
residual-only reset all remained below the strongest control.  Retained
population experience did not compound into this primitive.

## Verdict and next architecture gate

Do **not** promote this configuration to a longer run.  Complete replacement
policy reset would erase the old memory-utility policy, violating the project's
retention objective even though the sensory/recurrent controller remains
intact.

The next high-ROI experiment is architectural but tightly localized: add a
zero-initialized, context-retrieved residual capable of adjusting all generic
replacement features, not only the three newest features.  It must:

1. be a mathematical no-op for old contexts;
2. learn the redundancy atom without resetting the shared old policy;
3. leave old age/frequency/reliability behavior bit-identical when the new
   residual is absent;
4. fail under reward alignment and novelty-feature shuffles.

Only after that sub-minute gate passes should any three-minute run begin.

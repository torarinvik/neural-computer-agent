# Optional-thought transfer — pre-registration

## Question

Does the replicated 105-parameter attempted-action advantage learner transfer
to a genuinely new compute decision faster than a same-architecture reset
learner?

The new operation is **answer now versus perform one additional recurrent
thought step and then answer**. It is not an external-memory read. The
controller, sensory encoder, working state, and persistent memory are frozen.
Only the tiny compute-allocation head learns.

## Why this is a valid next rung

A preflight over 16,384 unseen four-rule dot-pair lifetimes found:

- extra thought helped 301 (`1.837%`);
- extra thought harmed 106 (`0.647%`);
- it changed nothing on 15,977.

A discarded diagnostic MLP predicted help versus harm at `79.25%` held-out
accuracy from four generic learner-visible statistics: immediate confidence,
top-two margin, action entropy, and controller hidden-state change. This proves
that the decision is not random, but the diagnostic weights and private labels
are not used by either experimental learner.

Because naturally sampled cases are overwhelmingly neutral, the verifier
privately screens fresh lifetimes and selects equal numbers where thought helps
and harms. This is curriculum construction, not learner input. Every screened
lifetime and both counterfactual verifier bits are accounted separately.

## Learner-visible information

For each selected lifetime the learner receives:

- the four generic controller statistics;
- an opaque uniformly attempted action (answer or think);
- its exact logging propensity, `0.5`;
- only the attempted action's scalar verified outcome;
- a generic thought cost of `0.01`.

It never receives the unattempted outcome, correct compute action, curriculum
help/harm label, semantic task identity, or correct answer. Its target is the
same inverse-propensity attempted-action advantage used by the successful
external-read allocator.

## Sub-minute horse race

- inherited arm: reconstructed seed-7425 width-16 advantage weights;
- reset arm: identical 105-parameter architecture with fresh weights;
- reward-shuffled, feature-shuffled, and zero-evidence controls;
- 720 selected training lifetimes/bits, 12 optimizer updates, zero replay;
- a private balanced 256-lifetime both-action audit;
- metrics every 120 selected bits.

The inherited weights were originally learned on optional external-memory
reading. Reconstructing them deterministically is replay for artifact recovery,
not new scientific experience, and its historical 720-bit source budget remains
attached to the checkpoint.

## Pre-registered gates

The inherited arm must:

1. reach at least `65%` private compute-choice accuracy;
2. beat the strongest fixed policy by at least `0.10` verified utility;
3. capture at least `20%` of the available oracle gap;
4. cross and permanently retain all three thresholds using strictly fewer
   selected verifier bits than reset (or while reset never crosses);
5. lose at least `0.05` utility under each causal control;
6. lose at least `0.05` utility when held-out evidence is shuffled;
7. retain binary mapping and four-rule behavior;
8. preserve live gradients and exact save/reload.

One passing seed permits an unchanged unseen-seed replication. One failure is
localized; training time is not increased unless the learning curve provides
specific high-ROI evidence.

The main claim is bits-to-threshold transfer, not final accuracy. Private
screening cost, learner-visible experience, counterfactual audit bits, optimizer
updates, replay, and wall time are reported separately.

## Seed 7801 result and one-axis localization

Direct whole-head transfer failed. The inherited arm chose thought on `100%`
of cases from 0 through 600 bits. It finally became causal at 720 bits
(`57.8%` choice accuracy and `+0.071` utility), but did not pass. The reset
arm found a smaller signal earlier. No longer run or replication is justified.

Seed 7802 changes exactly one transfer boundary: retain the inherited
LayerNorm and hidden feature extractor, but reset the one-neuron decision
output to zero. This tests whether reusable representation was masked by the
old operation-specific output bias. The direct whole-head arm remains in the
report as a bounded diagnostic. Controls now share the trunk-transfer
initialization, and all budgets and gates remain unchanged. A failure closes
this immediate transfer fork; it will not trigger more experience.

## Final result

Seed 7802 rejected trunk-only transfer:

- inherited trunk: `57.4%` choice accuracy, `+0.0683` utility, `13.8%`
  oracle-gap capture;
- matched reset: `59.0%`, `+0.0853`, `17.2%`;
- intact inherited head: `64.1%`, `+0.1334`, `26.9%`, but below the
  pre-registered choice gate and without a stable crossing;
- binary and four-rule retention, gradients, persistence, and the evidence
  shuffle passed;
- the trunk arm failed the main performance, speed, and reward-control gates.

Both runs used 720 learner-visible verifier bits and 12 updates. Each screened
86,016 logical lifetimes and charged 172,032 private curriculum-verifier bits;
this high selection cost is another reason not to scale the present task.

The attempted-action advantage **objective and architecture blueprint**
transfer, but these operation-specific weights do not yet provide reliable
positive forward transfer. The next high-ROI atom is to align a shared
compute-value representation across multiple optional operations during
training, while retaining operation-specific output adapters. It must race
against matched reset learners on a third held-out operation. No experiment was
left running.

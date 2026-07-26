# Capacity-five compounding rung — pre-registration

## Question

Does the compute-allocation knowledge learned at memory capacity three and
verified at capacity four immediately transfer to capacity five, or at least
reduce stable bits-to-threshold versus an identical reset learner?

This changes exactly one environmental axis. The sensory stream, controller,
four generic read statistics, attempted-action advantage objective, read cost,
optimizer, and training budget remain unchanged. The learner is not told the
capacity.

## Powered sub-minute race

- inherited replicated 105-parameter advantage head versus matched reset;
- reward-shuffled, feature-shuffled, and missing-evidence controls;
- 720 fresh learner-visible attempted-action outcomes;
- 12 optimizer updates and zero replay;
- 2,040 held-out contexts, giving 4,080 private counterfactual audit bits;
- metrics every 120 training bits.

The inherited arm must:

1. maintain at least `65%` compute-choice accuracy;
2. beat the strongest fixed choice by at least `0.05` verified utility;
3. capture at least `20%` of the oracle gap;
4. cross all three thresholds stably before reset;
5. lose at least `0.02` utility under every causal control;
6. retain binary mapping and four-rule ability;
7. preserve live gradients and exact save/reload.

Seed 7831 is the discovery run. A complete pass permits exactly one unchanged
seed-7832 replication. Failure does not earn a larger budget; it is localized
before another mechanism or curriculum change.

Transfer distance itself creates no reward. It increases the value of a result
only when stable verified capability, positive transfer over reset, retention,
and causal evidence all pass.

## Seed 7831 result and corrected compounding test

Direct capacity-three inheritance did not pass capacity five: it crossed
stably at 240 bits while reset crossed at 120. Final inherited capability was
real and causal (`68.7%` choice accuracy and `44.9%` gap capture), but it was
not more sample-efficient than reset. No replication or larger budget is
allowed.

This exposed a lineage omission. The successful capacity-four experiment
evaluated transfer but never saved the weights after learning capacity four.
Testing those original capacity-three weights again at capacity five measures
repeated zero-shot reach, not sequential compounding.

The corrected experiment therefore:

1. reconstructs the capacity-four learning phase and saves its final allocator;
2. races that consolidated allocator on capacity five against both the
   original capacity-three allocator and a matched reset learner;
3. requires stable mastery strictly earlier than **both** baselines;
4. keeps the same 720-bit capacity-five budget, powered audit, controls, and
   retention gates;
5. accounts the 720 capacity-four source bits separately in checkpoint
   provenance.

Seed 7841 is the discovery run. Only a complete pass permits one unchanged
seed-7842 replication.

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

## Replicated sequential-compounding result

The capacity-four consolidation was reconstructed exactly from seed 7824. It
passed every original powered gate and saved a checkpoint whose provenance
records 720 source verifier bits and 12 source updates.

Capacity five then passed twice:

- seed 7841: consolidated lineage stable at `0` new bits; original
  capacity-three lineage and reset both at `120`; final consolidated accuracy
  `74.8%` and oracle-gap capture `53.4%`;
- seed 7842: consolidated lineage stable at `0`; original capacity-three
  lineage at `240`; reset never stable within `720`; final consolidated
  accuracy `73.8%` and gap capture `53.5%`.

Every speed, fixed-utility, oracle-gap, control, gradient, persistence, binary
retention, and four-rule retention gate passed. The capacity-five learner saw
720 fresh verifier bits per arm and zero replay. The intermediate
capacity-four checkpoint separately accounts its 720 source bits.

This is stronger than repeated zero-shot transfer: when capacity-four
experience was not consolidated, capacity-three inheritance lost to reset at
capacity five. After consolidating the intermediate experience, the same
architecture immediately mastered capacity five and beat both the older
lineage and reset on two target streams.

The next frontier is to keep capacity five fixed and change one statistics
distribution axis, such as occupancy or retrieval reliability. That begins
testing abstraction beyond pure bank-size scaling while preserving the
verified compounding chain.

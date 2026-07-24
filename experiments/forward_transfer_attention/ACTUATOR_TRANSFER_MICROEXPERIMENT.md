# Zero-label actuator-transfer microexperiment

## Question

Does a latent intention learned from visual experience and attempted-action
success reduce the number of new reward bits needed to control an unfamiliar
action protocol?

This is the first device-independence rung. It is not yet evidence that one
cognitive primitive accelerates acquisition of a different cognitive
primitive.

## Learner-visible information

- rendered RGB support stream;
- its recurrent latent state;
- its own attempted action;
- the known uniform logging propensity;
- the resulting scalar verifier reward.

The learner never receives the first/last rule, correct action, object identity,
palette identity, task ID, or a target for an action it did not attempt.

## Two phases

1. **Acquire an intention.** Train an eight-dimensional bottleneck and a
   two-action success decoder on the support-order task. Training uses only BCE
   on the attempted action's observed reward.
2. **Calibrate a new device.** Freeze the acquired intention module, discard
   the old decoder, and train a fresh four-command adapter. A seed-specific
   protocol maps the two latent intentions onto two of four command IDs. The
   other two commands are distractors. The mapping is learned only through
   attempted commands and scalar outcomes.

## Equal-accounting arms

- experienced intention plus a fresh adapter;
- an identical fully fresh intention-plus-adapter learner;
- experienced intentions shuffled across logical lifetimes;
- attempted commands shuffled relative to experience;
- observed rewards shuffled relative to experience.

Every point reports unique phase-B reward bits, optimizer updates, examples
processed, wall time, and held-out verified accuracy.

## Primary metrics

- phase-B reward AULC above the 50% verified-accuracy majority floor;
- unique phase-B reward bits to 55%, 65%, and 75% held-out accuracy;
- experienced/fresh reward-bit ratio at every crossed threshold.

Final accuracy alone cannot establish faster learning.

Uniform exploration earns reward on 25% of interactions because it samples
four commands. Verified argmax accuracy has a stricter 50% majority floor,
because only the two protocol-mapped commands can be correct. These two
baselines must not be conflated.

## Causal audits

- **True support reversal:** re-render reversed event order while preserving the
  rewarded identity. The correct intention and device command must change.
- **Stale intention:** pair every held-out episode with an intention extracted
  from an episode carrying the opposite private rule. Private rule metadata is
  used only by this offline audit, never by the learner. A naive one-step roll
  is also reported for continuity, but is not the gate because balanced
  generator ordering can preserve the same rule more than half the time.
- **Protocol swap:** exchange the device codes assigned to the two intentions.
  The unchanged adapter should fail predictably; this shows the adapter, rather
  than the reasoner, owns protocol semantics.
- **Held-out palette:** training and evaluation use disjoint color-pair sets.

Hidden recurrent-state swaps are forbidden because they create sequences the
controller could never have produced.

## Promotion rule

One seed is provisional. Advance to two more seeds only if the experienced arm:

- exceeds the best valid control by at least 0.03 AULC;
- reaches at least 60% held-out accuracy;
- reaches a threshold with fewer reward bits than the fresh learner;
- scores at least 60% on true support reversal with at least 50% command flips;
- falls toward the 50% majority floor under stale-intention evaluation.

Only after all three seeds pass may this be recorded as zero-label actuator
transfer. A different-primitive experiment remains required for compounding
cognitive learning.

## Three-seed result (2026-07-24)

All three seeds passed the pre-registered v2 gate. At an equal 200 optimizer
updates and 6,000 replayed examples per reward-bit prefix:

| Metric | Seed 211 | Seed 257 | Seed 313 | Mean |
|---|---:|---:|---:|---:|
| Experienced final accuracy | 77.60% | 81.51% | 81.77% | 80.30% |
| Fresh final accuracy | 77.08% | 79.17% | 78.13% | 78.13% |
| Experienced AULC above 50% | 0.2828 | 0.3073 | 0.3151 | 0.3017 |
| Fresh AULC above 50% | 0.1750 | 0.2115 | 0.2010 | 0.1958 |
| Reward bits to 75%, experienced | 32 | 32 | 32 | 32 |
| Reward bits to 75%, fresh | 510 | 256 | 256 | 340.7 |
| True reversal accuracy | 77.08% | 80.47% | 81.51% | 79.69% |
| True reversal command flips | 54.69% | 61.98% | 63.28% | 59.98% |
| Opposite-rule stale accuracy | 22.40% | 18.49% | 18.23% | 19.70% |
| Swapped-protocol accuracy | 22.40% | 18.49% | 18.23% | 19.70% |

The per-seed reward-bit ratios to 75% were 15.94×, 8×, and 8× (median 8×,
mean 10.65×). This is interaction efficiency at a fixed replay-compute budget,
not free learning: every curve point still consumed 200 updates and 6,000
processed examples.

The original one-step rolled-stale audit was retained but removed as a gate
after measurement showed that the generator ordering paired the same private
rule 53.65% of the time. The stronger audit guarantees an opposite-rule
intention for every episode and collapsed accuracy to 19.70% on average.

The supported claim is narrow but real: a latent intention acquired solely
from pixels, attempted actions, and scalar outcomes can be frozen and reused
to calibrate a new four-command device much faster than an identical fresh
system. This establishes zero-label actuator/interface transfer. It does not
yet establish transfer between different cognitive primitives or compounding
learning.

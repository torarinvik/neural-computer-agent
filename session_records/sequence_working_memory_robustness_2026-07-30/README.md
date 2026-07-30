# Position-invariant, distractor-resistant working memory

## Breakthrough

The same compact controller now conditionally reproduces or reverses a
two-event visual sequence across every tested position and across zero, one,
or two irrelevant intervening events.

The learner receives only RGB frames, its own opaque attempted actions, and
scalar outcomes. Sequence identities, requested operations, correct actions,
positions, and distractor counts are never learner inputs.

## What failed first

The previous specialist scored 99.16% at its training positions but only
11.08% on the shifted layout. Intermediate one-pixel shifts were even worse
(14.27% and 0.81%), revealing stride-phase aliasing in perception rather than
a working-memory failure.

Trying to bolt invariance onto the specialist was rejected:

- 32 updates at the nearest shifted position damaged base accuracy to 70.46%
  and left shifted accuracy at chance;
- 32 abrupt full-shift updates damaged base accuracy to 71.48% and also left
  shifted accuracy at chance;
- a 64-update gradual blend retained only 72.41% base and 53.03% shifted.

The high-ROI fix was data design: train the forward-retention primitive from
the beginning on balanced base, intermediate, and shifted positions. Position
is still only a nuisance pixel transformation; no coordinate reaches the
controller.

## Invariant retention

At seed 26301, a fresh two-item forward-span controller trained for 128 updates
(16,384 verifier bits) reached 100% at the base position and at every
intermediate/shifted position. A second seed required 64 more low-rate updates
to cross the same 100% gate, preserving the project's phase-transition
patience rule.

## Compounding manipulation learning

Starting mixed forward/reverse learning from the invariant forward skill
produces a large replicated sample-efficiency gain after only 32 updates
(4,096 new verifier bits):

| Seed | Retention parent | Fresh learner |
| --- | ---: | ---: |
| 26301 | 93.70% | 53.99% |
| 26501 | 91.55% | 44.73% |

The seed-26301 parent remains at 93.73% across every tested position after the
full mixed phase. The matched fresh learner does not cross 90% at 8,192 new
bits and reaches the same 93.73% plateau only at 16,384 bits. At the measured
thresholds, prior retention therefore provides a 4x new-skill
sample-efficiency gain.

Both lineages retain forward reproduction at approximately 100%; the remaining
error is in reversal. This is reuse rather than a trade that sacrifices the
old primitive.

## Distractor resistance

Before distractor experience, the invariant working-memory controller scores
68.68% with one X-shaped irrelevant event. After 64 updates alternating zero-
and one-distractor batches:

| Audit | Seed 26401 | Seed 26501 |
| --- | ---: | ---: |
| zero distractors | 93.71% | 93.48% |
| one distractor | 93.71% | 93.48% |
| two unseen distractors | 93.58% | 93.48% |

Thus the controller learns a useful timing-invariant retention policy rather
than memorizing exactly one delay. The selected checkpoint remains above 93%
at every position with one or two distractors.

The capacity boundary is honest and sharp:

- four distractors: 82.12%;
- eight distractors: 67.26%.

## Adversarial controls

On the selected one-distractor checkpoint:

- blank sequence evidence: 49.85%;
- operation cue blanked: 74.89%, the analytic operation-blind shortcut;
- complete fast-memory reset: 50.00%;
- workspace disabled: 77.51%;
- recurrent state reset while preserving workspace: 81.08%;
- valid operation reversal: 93.79% accuracy and 75.00% non-palindrome flips;
- shuffled outcomes during adaptation: 50.00% everywhere and zero flips.

The skill therefore depends on visual evidence, scalar verifier outcomes, and
RAM/VRAM-resident fast memory. Both recurrent state and differentiable
workspace are causal, partially redundant carriers.

## Rejected extra duration and endpoint curriculum

Extending mixed training from 128 to 256 updates did not move the 93.73%
held-out plateau. Error decomposition found one unresolved conjunction: one
reverse-operation/non-palindromic first output. Training reversal alone or one
output endpoint merely moved the error to another output and did not improve
total generalization. These forks are stopped.

The next frontier is not more duration. It is generic selective retention over
four or more distractors, followed by span three. If error remains after a
one-axis curriculum, a learned write/keep gate becomes justified.

## Artifacts

- `artifacts/checkpoints/unified_robust_sequence_working_memory_seed26401.pt`
- JSON reports in this directory preserve the baseline, rejected bolt-on,
  transfer controls, two replicas, distractor scaling, and shuffled outcomes.

This checkpoint is the promoted sequence-working-memory specialist. It is not
yet consolidated into the larger relation/magnitude/numerosity controller.

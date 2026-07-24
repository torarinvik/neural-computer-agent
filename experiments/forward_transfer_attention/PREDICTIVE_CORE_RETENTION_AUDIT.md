# Behavioral retention audit after sequential predictive training

## Question

Did the paired spatial predictive update erase or attenuate the earlier
temporal capability, thereby explaining why it failed to improve the third
same/different primitive?

Predictive loss and representation similarity are secondary diagnostics.
Verified behavior is the primary retention test.

## Compared cores

All begin from identical initial weights:

- temporal paired prediction only;
- temporal paired, then spatial paired prediction;
- temporal paired, then spatial future targets shuffled;
- temporal paired, then equal additional temporal prediction;
- fully fresh.

Each frozen core receives an identical newly initialized intention/action head.
Every head trains on the same balanced temporal logged buffer, 510 observed
reward bits, 200 optimizer updates, and 6,000 examples. Training uses only the
attempted action's observed reward.

## Audits

- normal held-out support-order accuracy;
- true support-order reversal with selected identity preserved;
- prediction flip rate;
- fully fresh floor.

## Interpretation

- A paired-spatial drop of at least three accuracy points relative to the
  temporal-only core, reproduced in reversal accuracy, is provisional evidence
  of behavioral forgetting and promotes a rehearsal experiment.
- Stable temporal retention but weaker third-task transfer means spatial
  experience was redundant or task-irrelevant rather than destructive.
- If all second-stage updates degrade retention, generic plasticity loss is
  more likely than spatial-specific interference.

One seed localizes only. A load-bearing forgetting claim requires replication.

## Seed-211 result

The paired spatial update caused a suggestive but sub-threshold decline:

| Core | Normal temporal accuracy | Reversed accuracy | Reversal flips |
|---|---:|---:|---:|
| Temporal only | 77.08% | 76.82% | 53.91% |
| Temporal + paired spatial | 74.48% | 74.22% | 48.70% |
| Temporal + shuffled spatial | 75.52% | 76.56% | 52.08% |
| Temporal + extra temporal | 78.91% | 78.65% | 57.55% |
| Fully fresh | 50.00% | 50.00% | 0.00% |

The paired-spatial normal and reversal drops were both 2.60 points. The
pre-registered gate required at least three points, so no rehearsal experiment
was promoted. The result is consistent with modest spatial-specific drift but
is not strong enough to justify optimizing a retention mechanism.

The next higher-information experiment is closed-loop micro-intercept, where
action-conditioned prediction can be tested on consequences the agent
actually controls.

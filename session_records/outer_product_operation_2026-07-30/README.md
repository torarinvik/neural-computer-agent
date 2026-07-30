# Pairwise latent operation binding — 2026-07-30

## Breakthrough

The controller now binds the immediately preceding sensory event to an
inherited amodal intention through an explicit low-dimensional outer product.
Two learned rank-8 projections produce all 64 pairwise products; a successor
slot may read them alongside its ordinary latent inputs. The interface carries
no task ID, operation label, count, correct action, unattempted-action outcome,
or verifier-private value.

This is a replicated improvement in **history-free causal operation use**, not
90% mastery. The selected checkpoint remains explicitly unpromoted.

At 256 updates / 2,048 unique lifetimes:

| 12,288 scalar outcomes | Seed 25301 | Seed 25311 |
|---|---:|---:|
| sequential conditional operation | **85.10%** | **84.31%** |
| history-free normal cue | **86.03%** | **85.57%** |
| history-free reversed cue | **86.07%** | **85.24%** |
| paired history-free accuracy | **86.05%** | **85.41%** |
| cue-reversal prediction flips | **86.47%** | **82.04%** |
| timing-matched blank cue | 49.70% | 49.77% |
| paired blank cue | exactly 50% | exactly 50% |
| relation retention | 99.19% | 99.09% |
| magnitude retention | 91.30% | 92.68% |
| numerosity retention | 90.20% | 90.68% |

The preceding one-event-RAM architecture reached 84.82% sequential, 84.58%
history-free, and 82.25% cue-reversal flips on seed 25301. The pairwise
candidate therefore adds +0.29 sequential points, +1.47 history-free points,
and +4.22 flip-rate points at the same experience and optimizer budget.

The prospective seed-25311 replication is more diagnostic: at 128 updates the
old elementwise interaction reached 73.52% history-free with 53.58% flips,
while the pairwise interaction reached 79.87% history-free with 68.18% flips.
At 256 updates it crosses 85% history-free in both cue directions while
preserving the inherited repertoire.

## Mechanism

The earlier interaction required two learned projections to discover aligned
coordinates before multiplying them element by element:

```text
tanh(W_event × previous_event) * inherited_intention
```

The new interface exposes every interaction between two compact learned
projections:

```text
event_8 = tanh(W_event × previous_event)
intent_8 = tanh(W_intent × inherited_intention)
pairwise_64 = flatten(event_8 outer_product intent_8)
```

The controller remains responsible for discovering what those coordinates
mean. The outer product supplies a generic relational bias rather than a
hand-written operation.

The appended slot is still zero-output initialized. Insertion therefore
preserves every old logit exactly, and all inherited weights remain frozen.
No additional controller step is added.

## Causal controls

All controls retain the same sensory protocol and scalar attempted-action
feedback:

- matched rank-8 module with only outer-product content zeroed:
  82.56% sequential, 61.95% history-free, 29.53% flips;
- shuffled verifier outcomes: 51.91% sequential, 52.04% history-free,
  13.69% flips;
- blank public cue: exactly 50% paired accuracy and zero flips;
- cue-only counterfactual: count pixels remain bit-identical while every
  correct answer is complemented;
- all three inherited behavioral gates remain above 90%.

Widths 4 and 12 reached only 83.55% and 83.08% sequential accuracy at the
matched 128-update discovery budget. Width 8 was selected before replication.
Adding the old elementwise interaction alongside the outer product was also
worse than the outer product alone.

## Error localization

A separate 8,192-lifetime read-only audit compared the selected controller
with its frozen numerosity parent on exactly the same clean count frames:

- inherited relation accuracy: 91.03%;
- outer-product controller, independent events: 86.49%;
- outer-product controller, sequential query suffix: 85.53%;
- 38.25% of its remaining errors coincide with an inherited relation error;
- when the parent relation is correct, conditional accuracy is 90.18%;
- action zero remains the weakest branch at 82.04%.

The next bottleneck is therefore shared between imperfect inherited
numerosity and an asymmetric action-zero decision boundary. It is no longer
missing temporal context or insufficient pairwise binding.

## Rejected small forks

- exact uniform action exploration reduced sequential accuracy to 82.63%;
- generic residual-norm prices did not beat the unpriced learner;
- a 20%→24.8% appearance curriculum improved the ordinary sequential screen
  slightly but weakened the stronger history-free counterfactual audit;
- advancing the inherited numerosity frontier from 24.8% to 26.2–26.3%
  failed under both eight- and sixteen-lifetime continuations;
- a latent-read-only outer-product slot did not dominate the ordinary slot;
- suppressing irrelevant previous reward at inference gained only 0.1–0.5
  points;
- extending the pairwise learner from 128 to 256 updates improved
  history-free causality, but sequential accuracy remained near 85%, so
  further duration scaling is stopped.

These are bounded negatives in the measured regime.

## Accounting and artifact

Each 256-update run uses:

- 2,048 unique logical lifetimes;
- 12,288 scalar attempted-action outcomes;
- 24,576 sensory frames;
- two sensory steps per answer;
- no replay of new-task events;
- no semantic or hand-labeled training target.

Selected research checkpoint:

`artifacts/checkpoints/unified_outer_product_operation_candidate_seed25301.pt`

SHA-256:

`bc915c4e2fef31172958e17808228363587c9127e981809d3dc22018a7769693`

The checkpoint is a load-bearing research candidate but is not an admissible
parent for the promoted lineage until sequential and history-free accuracy
both reach 90% on prospective seeds.

## Frontier

The rank-8 outer product is the next verified architectural breakthrough:
explicit pairwise latent structure improves causal operation generalization
across two seeds without additional sensory time or forgetting.

The remaining frontier is to raise the shared action-zero branch rather than
add binding capacity or duration. Any next intervention must improve both
sequential and history-free accuracy, retain the three inherited skills, and
survive blank-cue, shuffled-outcome, outer-content, and cue-reversal controls.

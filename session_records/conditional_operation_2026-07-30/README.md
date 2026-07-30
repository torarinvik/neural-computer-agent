# Conditional-operation audit — 2026-07-30

## Result

The controller learned to choose **larger or smaller on demand** from a public
visual operation cue while the count scene remained otherwise identical. This
closes the fixed-operation shortcut in the earlier
`visible_pair_numerosity_smaller` experiment.

This is a verified intermediate competence, not promoted mastery. The selected
checkpoint is deliberately marked
`unpromoted-causal-operation-research-candidate` because it has not reached the
90% promotion threshold.

| 256 updates / 2,048 unique lifetimes | Seed 25231 | Seed 25261 |
|---|---:|---:|
| sequential held-out accuracy | 81.08% | 79.68% |
| timing-matched blank cue | 50.18% | 49.82% |
| relation retention | 99.20% | 99.34% |
| magnitude retention | 90.42% | 91.29% |
| numerosity retention | 88.23% | 88.25% |
| history-free cue counterfactual accuracy | 70.26% | 67.27% |
| history-free prediction flips | 46.25% | 40.11% |
| paired blank-cue accuracy | exactly 50% | exactly 50% |
| paired blank-cue prediction flips | 0% | 0% |

Each selected run consumed 12,288 scalar verifier outcomes and 24,576 sensory
frames. The model received no operation ID, count, correct action,
unattempted-action outcome, semantic label, or task-state hook.

## The audit corrected an earlier false claim

The previous fixed-smaller task scored 83.96% with its visual cue. Its original
missing-cue control also removed the entire prestimulus timestep, reducing
accuracy to 40.21%. A timing-matched control retained that extra frame but
blanked only its cue pixels. Accuracy remained **81.93%**. Therefore the model
had mostly learned “the extra-frame task means smaller”; it had not established
visual operation-cue use. That claim is retracted.

The corrected task samples one larger and one smaller request in every
two-event pair, and three of each in every six-event lifetime. Operation is
balanced independently of the count relation. A fixed larger or smaller policy
is chance. `reverse_operations=True` preserves every count pixel, complements
only the public cue, and flips every verifier answer.

## What made the corrected task learnable

The frozen controller already contains both required ingredients:

- its inherited amodal intention carries the larger-count decision;
- a disposable diagnostic probe decoded the operation cue from the post-cue
  recurrent state at 100% held-out accuracy.

The new generic binder appends
`tanh(W · recurrent_state) × inherited_intention` to the new skill slot's
latent inputs. It contains no semantic feature or verifier-private value and is
exactly behavior preserving at insertion because the slot's output remains
zero initialized.

At a matched 128-update budget and seed:

| Interface | Held-out | History-free flips |
|---|---:|---:|
| concatenation only | 71.60% | 13.11% |
| multiplicative latent binding | 75.76% | not recorded in the early pilot |

The 256-update product run reached 81.08%. Extending it to 512 updates reached
only 82.60%, so duration scaling was stopped as low return.

## Adversarial controls

At the 128-update discovery budget:

- shuffled verifier outcomes: **49.87%**;
- inherited latent/intention content ablated with identical model shape:
  **49.78%**;
- timing-matched blank cue: **50.05%** in the truthful product run;
- paired history-free blank replay: exactly **50%**, because every public
  tensor is identical while every verifier answer is complemented.

The controls show that acquisition requires all three of:

1. the visual operation signal;
2. the genuine sensory–outcome relationship;
3. inherited cognitive content.

## Frontier

The controller now has causal, reward-grounded conditional-operation
competence, but not mastery. Its sequential score is ~80%, while history-free
events are 67–70%. The next frontier is to make the current event's operation
cue dominate without relying on recurrent episode context, then cross 90%
without reducing the inherited numerosity floor.

The selected local checkpoint is
`artifacts/checkpoints/unified_conditional_operation_candidate_seed25231.pt`.
It is a research candidate, not an admissible parent.

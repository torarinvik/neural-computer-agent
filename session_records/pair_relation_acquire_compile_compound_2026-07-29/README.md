# Acquire, compile, compound: third relation appearance

## Breakthrough

The same unified controller now applies one learned same/different relation to
bars, diamonds, and disconnected dot pairs.  The promoted controller learned
the third appearance from 1,792 new lifetimes and passed an independent
8,192-lifetime audit:

| appearance | accuracy | complete causal gate |
|---|---:|:---:|
| bars | 99.96% | pass |
| diamonds | 97.83% | pass |
| dot pairs | 96.44% | pass |

Removing the second object reduced accuracy to 49.60%, 50.05%, and 49.90%,
respectively.  Blank-vision, valid pixel-counterfactual accuracy, and
prediction-flip gates also passed on every appearance.  The inherited binary
mapping, visible context, and visible-context XOR skills scored 96.25%,
91.85%, and 90.90% in the promoted training report.

The curated checkpoint is
`artifacts/checkpoints/unified_pair_relation_three_appearance_seed9622.pt`,
SHA-256
`6dee3d9545f537d041edfe4e7a29df579f41be2b50eae8740d1c06318998ba4e`.

## Acquire

Eight permissive acquisition updates expanded the inherited relation residual.
Forty-eight consolidation updates then froze that residual and trained only its
existing nonlinear gate refiner using opaque self-distillation and rehearsal.
No semantic relation label, task ID, correct unattempted action, or hidden game
state entered training.

- optimizer updates: 56;
- new dot-pair lifetimes: 1,792;
- new verifier bits: 10,752;
- five replay streams: 896 lifetimes each;
- total verifier bits including replay: 37,632;
- measured end-to-end training/evaluation time: 9.67 seconds.

The preceding diamond bridge used 10,240 new lifetimes and 184,320 total
verifier bits.  This rung therefore used 5.71 times fewer new lifetimes and
4.90 times fewer total verifier outcomes.

## Compound

The same 1,792 dot-pair lifetimes were given to the two-appearance parent and
the earlier bars-only parent.  The two-appearance lineage won on every paired
seed:

| seed | bars+diamonds parent | bars-only parent | advantage |
|---:|---:|---:|---:|
| 9622 | 96.39% | 86.73% | +9.66 pp |
| 9631 | 93.03% | 86.75% | +6.29 pp |
| 9632 | 94.17% | 86.75% | +7.42 pp |
| mean | 94.53% | 86.74% | +7.79 pp |

This is a lineage-level transfer claim: the diamond-trained parent also
inherits a trained refiner, so the experiment establishes compounding reuse of
the acquired controller state as a whole, not that a single semantic variable
alone caused the gain.

A matched reset-slot controller used the two-appearance architecture, seeds,
training data, and budget.  It reached 93.96% on dots but retained only 85.25%
bars and 70.61% diamonds.  Capacity can learn part of the new task, but the
inherited relation state is causally necessary for the combined repertoire.

## Compile

The compute audit adds optional recurrent passes over the same pixels without
new actions, outcomes, or verifier information.  The promoted controller
already masters all three appearances with zero optional passes: one
controller step per sensory event, the physical minimum in this architecture.

| extra thought passes | bars | diamonds | dot pairs |
|---:|---:|---:|---:|
| 0 | 99.98% | 97.85% | 96.43% |
| 1 | 99.82% | 98.23% | 96.75% |
| 2 | 99.87% | 97.40% | 97.46% |
| 4 | 99.52% | 97.28% | 97.61% |
| 8 | 98.62% | 97.57% | 98.48% |

Extra thought trades a small dot-pair gain for lower bars performance and
higher compute.  The accuracy-first compiler therefore correctly selects zero
extra thought.

## False promotion caught

The first apparent 95% dot-pair candidate rehearsed bars and unrelated skills
but omitted diamonds.  A full-repertoire audit exposed diamond accuracy at
82.54%, so it was rejected.  The trainer now:

1. explicitly lists every inherited appearance;
2. rehearses all inherited forms of the relation;
3. excludes all relation forms from the unrelated-skill locality penalty;
4. requires every inherited appearance to pass before saving a checkpoint.

With full relation rehearsal, consolidation retention weight `4.0` recovered
diamonds to 97.84% while increasing dots to 96.39%.

## Honest boundary and next frontier

The fixed 56-update recipe is not a robust stable-bits threshold: the two
replication seeds reached 93.03% and 94.17% rather than 95%.  Population
selection or validation-triggered stopping is justified by the measured
non-monotonic learning curves, but its search experience must be accounted
separately.

The next dogfooding rung should alternate axes: add a new relation such as
larger/smaller on familiar contours, then test whether the three-appearance
controller makes that primitive or its next appearance cheaper to acquire.

# Gradual magnitude bridge produces forward transfer

## Breakthrough

The causal bars-magnitude controller learned its first changed contour rung
from only 256 new lifetimes, retained every earlier skill, and made the next
untrained contour rung solvable with zero additional experience.

The learner received rendered pixels, its own opaque attempted actions, scalar
verifier outcomes, and opaque frozen-controller rehearsal. It received no task
ID, larger/smaller word, correct unattempted action, relation bit, curriculum
stage label, or hand-labelled intermediate target.

The promoted checkpoint is hosted as
`checkpoints/unified_pair_magnitude_gradual_bridge_seed21515.pt` in the
`torarin87/neural-computer-agent` Hugging Face repository.

- SHA-256:
  `594f9f45b99c3d6d78536d2d0d1af40cb988e62cb6d128a774a95496ffb4f392`
- Parent:
  `unified_pair_magnitude_compound_seed21475.pt`
- Parameters: 388,191
- New experience: 256 unique lifetimes / 1,536 verifier bits
- Retention replay: 224 lifetimes across seven complete streams
- Total: 480 lifetimes / 2,880 verifier bits
- Optimizer updates: 8
- Measured training plus full internal evaluation: 13.9–14.5 seconds

## The difficulty axis was audited before training

The first morph implementation was wrong. Magnitude resizing thresholded the
already blended mask, so every nonzero blend became the same union contour.
The fix resizes the binary bar and diamond endpoints first, then interpolates
their rendered levels. Endpoint pixels remain exact and intermediate contours
are genuinely continuous.

The promoted parent’s zero-shot curve then identified the first real boundary:

| bars→diamonds blend | parent accuracy | causal gate |
|---:|---:|---:|
| 0% | 92.04% | pass |
| 6.25% | 91.80% | pass |
| 12.5% | 90.62% | pass |
| 14.0625% | 90.30% | pass |
| 15.625% | 89.47% | fail |
| 18.75% | 87.78% | fail |
| 25% | 83.92% | fail |
| 100% | 59.63% | fail |

Training therefore began at 14.0625% and ended at 15.625%. The task was not
made arbitrarily easy; it was set to the first measurable point beyond current
ability.

## The correct plasticity boundary

Editing the mastered magnitude slot was rejected. Eight updates left the 25%
target at 83.61% and damaged bars to 83.89%.

The successful architecture freezes every mastered parameter and appends one
zero-output 64-unit bridge slot. It reads only the immediately preceding
magnitude slot. The reset control receives the identical appended capacity but
has that inherited magnitude slot reset before training.

At the 25% rung, the appended experienced arm improved to 87.14% while reset
remained at 53.52%; doubling experience plateaued, so the jump was correctly
diagnosed as too large rather than rewarded with a long run.

At 15.625%, four updates (128 new lifetimes) first crossed mastery but were
seed-sensitive. A locality price of 0.1 stopped the new slot from opening on
unrelated XOR events; 1.0 suppressed acquisition and was rejected. Halving the
learning rate from 0.01 to 0.005 removed the remaining optimizer instability.

## Replicated acquisition

The final fixed recipe passed every capability, causal, and retention gate on
three of three seeds:

| seed | target | bars retained | delete object 2 | inherited-read advantage |
|---:|---:|---:|---:|---:|
| 21513 | 90.37% | 90.46% | 60.75% | +11.93 pp |
| 21515 | 91.22% | 91.72% | 60.79% | +11.74 pp |
| 21516 | 90.65% | 91.22% | 60.36% | +12.59 pp |

The exact matched reset recipe scored 52.60% on the target and 51.15% on bars.
The gain therefore comes from inherited knowledge, not the added slot alone.

## Independent causal audit

The promoted seed-21515 checkpoint passed a 16,384-lifetime disjoint audit:

- trained 15.625% contour: 91.36% normal, 91.21% reversed;
- pixel-counterfactual prediction flip: 82.93%;
- delete the second object: 60.52%;
- disable inherited latent reads: 79.20%, a 12.16-point loss;
- original bars magnitude: 91.68%;
- full diamonds: 60.58%.

Retained scores:

- pair relation: 99.21% bars, 96.56% diamonds, 97.59% dot pairs;
- binary mapping: 91.50%;
- visible context: 91.31%;
- visible-context XOR: 91.98%.

Blank vision, valid rendered reversal, object deletion, inherited-read
ablation, and all old-skill gates passed. The audit trained no parameter.

## The compounding gain reaches the next task

A second paired 16,384-lifetime audit gave parent and child identical fresh
events:

| unseen blend | parent | child | child gate | gain |
|---:|---:|---:|:---:|---:|
| 15.625% | 89.45% | 91.21% | pass | +1.76 pp |
| 17.1875% | 88.57% | 90.68% | pass | +2.11 pp |
| 18.75% | 87.87% | 90.31% | pass | +2.44 pp |
| 20.3125% | 87.08% | 89.90% | fail | +2.82 pp |
| 21.875% | 86.17% | 89.11% | fail | +2.94 pp |
| 23.4375% | 85.33% | 88.47% | fail | +3.13 pp |
| 25% | 84.17% | 87.69% | fail | +3.51 pp |

The child masters 17.1875% and 18.75% without training on either. This closes
the project’s compounding loop for this rung: learning one primitive variation
reduces the evidence required for the next variation all the way to zero.

## Experience first, computation second

At the trained 15.625% contour:

| optional thought steps | accuracy |
|---:|---:|
| 0 | 91.30% |
| 1 | 90.39% |
| 2 | 88.54% |
| 4 | 87.05% |
| 8 | 85.50% |

The controller is already compiled to the physical minimum of one pass per
sensory event. Extra thought is overthinking.

## What worked and what failed

Worked:

- measuring the zero-shot difficulty curve before choosing a rung;
- exact continuous rendering with endpoint invariance tests;
- immutable mastered slots plus a zero-output successor slot;
- immediately preceding latent reads;
- complete seven-stream replay at only four lifetimes per stream;
- a small locality price applied only to interference surfaces;
- lower learning rate at the same experience budget;
- experienced-versus-reset causality and paired forward-transfer audits.

Rejected:

- thresholding before contour interpolation;
- editing the mastered magnitude slot;
- jumping directly to 25%;
- extending a flat 25% run merely by spending longer;
- no locality cost, which occasionally disturbed XOR;
- locality 1.0, which suppressed learning;
- learning rate 0.01, which was seed-sensitive at eight updates.

## Next frontier

The new controller already masters 18.75% without additional experience.
The next first-beyond-ability rung is therefore 20.3125%, not another
arbitrary jump. Start at 18.75%, train toward 20.3125%, compare inherited and
reset immediate magnitude slots, and retain every current gate. Only escalate
past 256 new lifetimes if the sub-minute curve shows positive learning.


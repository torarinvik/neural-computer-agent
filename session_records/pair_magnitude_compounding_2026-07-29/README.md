# Causal magnitude compounding

## Breakthrough

One 369,926-parameter controller acquired a genuinely new visual relation:
compare two simultaneously visible objects and emit the opaque action
corresponding to which is larger. It retained its complete earlier repertoire,
causally reused the immediately preceding same/different relation, and already
executes at one controller pass per visual event.

The learner saw only rendered RGB frames, its own opaque attempted actions, and
scalar verifier outcomes. It received no task ID, semantic larger/smaller
label, correct unattempted action, hidden relation bit, or hand-labelled
intermediate target.

The promoted checkpoint is hosted at
`checkpoints/unified_pair_magnitude_compound_seed21475.pt` in
`torarin87/neural-computer-agent` on Hugging Face.

- SHA-256:
  `3717bc318a35c7508c6d9fe7be0b2a196b0883af99d9f08c7205f196c61a7dfa`
- Parent:
  `unified_pair_relation_robust_three_appearance_seed9672.pt`
- New-skill training: 4,096 updates and 131,072 unique lifetimes
- Balanced replay: 65,536 lifetimes
- Total verifier bits: 1,179,648
- Training plus internal evaluation: 272.45 seconds

## Why this is a real comparison

The first renderer was rejected. It used only one large and one small absolute
size; the learned controller reached 86.8%, but still scored 75.5% when the
second object was deleted. It had learned an absolute-size shortcut.

The corrected task uses five overlapping absolute size levels and samples only
adjacent pairs. Either object alone has a theoretical optimum of 62.5%; the
relation between both objects is the only deterministic solution. A frozen
state diagnostic reached 93.75% with both objects and 62.38% with the second
object removed.

The magnitude-only positions live in a separate renderer bank. The original
four-position banks used by every inherited task remain bit-identical.

## Cheap-to-expensive ladder

The retention repair did not add replay. The one same/different replay batch
already paid for each update now cycles through bars, diamonds, and disconnected
dot pairs.

| rung | updates | seconds | magnitude | bars relation | diamonds | dots |
|---:|---:|---:|---:|---:|---:|---:|
| micro | 512 | 35.22 | 66.53% | 96.74% | 97.92% | 90.54% |
| medium | 2,048 | 130.61 | 83.52% | 98.99% | 97.92% | 95.83% |
| graduation 21473 | 4,096 | 272.89 | 92.69% | pass | pass | pass |
| graduation 21474 | 4,096 | 275.31 | 90.14% | pass | pass | pass |
| graduation 21475 | 4,096 | 272.45 | 91.98% | pass | pass | pass |
| graduation 21476 | 4,096 | 275.00 | 89.96% | pass | pass | pass |

The 512-update rung moved the forgotten dot relation from the earlier 54.9%
failure to the mastery boundary in under one minute. That evidence justified
2,048 updates; complete retention there justified the 4,096-update population.
Three of four graduation seeds passed their internal mastery gate. Population
search compute is reported separately; only seed 21475 passed every independent
compounding gate and was promoted.

## Independent causal audits

The promoted checkpoint passed two disjoint audit streams:

| audit | lifetimes | magnitude | reversed | flip rate | remove object 2 | remove inherited read |
|---|---:|---:|---:|---:|---:|---:|
| seed 22475 | 8,192 | 92.05% | 91.97% | 84.27% | 60.53% | 81.34% |
| seed 32475 | 16,384 | 91.96% | 91.76% | 83.90% | 60.40% | 81.61% |

Disabling the inherited relation read therefore costs 10.70 and 10.34
percentage points. This is direct causal evidence that the earlier primitive
makes acquisition useful, rather than merely coexisting with the new slot.

Fresh retained scores on the larger audit were:

- pair relation: 99.49% bars, 97.71% diamonds, 95.48% dot pairs;
- binary mapping: 94.20%;
- visible context: 92.19%;
- visible-context XOR: 91.43%.

Blank vision returned to chance. Reversing the rendered size order while
holding nuisance variables fixed preserved accuracy and flipped predictions.
Removing one object, removing inherited latent content, and evaluating every
old skill were all performed without training any parameter.

Magnitude did not transfer to unseen diamond or dot-pair contours (about
60%). This is deliberately reported as the next gradual curriculum boundary,
not hidden inside the trained-bars claim.

## Experience first, thought second

After acquisition was fixed, an audit held the checkpoint and external
evidence constant and varied only optional recurrent thought:

| extra thought steps | normal accuracy |
|---:|---:|
| 0 | 91.92% |
| 1 | 91.33% |
| 2 | 89.53% |
| 4 | 88.20% |
| 8 | 87.22% |

Zero optional thought is both the most accurate and the physical minimum: one
controller pass per event. More thought becomes overthinking.

The resulting standing curriculum rule is:

1. minimize unique verifier experience to stable causal mastery;
2. require complete old-skill retention and positive inherited-read causality;
3. freeze the evidence stream and minimize optional thought/latency;
4. accuracy and retention are constraints, never terms that a speed reward may
   trade away.

## What failed and what worked

Worked:

- adversarial one-object ablation before promotion;
- five overlapping sizes that cap one-object shortcuts at 62.5%;
- moderate cosine decay from 0.02 to 0.005;
- one immediately preceding readable relation slot;
- cost-neutral full-repertoire replay;
- sub-minute → three-minute → graduation escalation;
- population selection followed by independent fresh-world audits.

Rejected or bounded:

- the original two-absolute-size renderer rewarded a shortcut;
- replaying only bars silently forgot dot pairs;
- the first accepted magnitude checkpoint used inherited information strongly
  but failed full relation retention;
- not every internally accepted population member passed the stricter
  independent compounding audit;
- extra recurrent thought reduced accuracy;
- cross-contour magnitude transfer remains unsolved.

## Next frontier

The next rung is a gradual appearance bridge for magnitude: bars first,
partially morphed contours next, then diamonds and disconnected dot pairs.
Admission requires lower experience-to-mastery than a reset-read control,
complete retention, and causal dependence on earlier knowledge. Only after each
appearance is mastered should optional thought be optimized.


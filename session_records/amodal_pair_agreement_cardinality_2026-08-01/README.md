# Pair-agreement cardinality scaling — 2026-08-01

## Breakthrough

The self-supervised pair-agreement router scales beyond the original N=3
experiment. The same frozen agreement heads and input bus were evaluated with
two useful complementary streams plus zero through three valid distractor
streams. Each event receives the strongest pair score it has with any other
event, thresholded at 0.4; no task or action information is used at routing
time.

| Cardinality | No agreement, seed 175101 | With agreement, seed 175101 | No agreement, seed 175201 | With agreement, seed 175201 |
|---:|---:|---:|---:|---:|
| N=2 | 96.61% | 96.61% | 96.61% | 96.61% |
| N=3 | 58.29% | 95.98% | 58.29% | 95.90% |
| N=4 | 56.01% | 94.32% | 56.01% | 94.03% |
| N=5 | 53.43% | 91.41% | 53.43% | 90.65% |

Both seeds pass the cardinality gate: N=2 ≥90%, N=3 through N=5 ≥85%, and
each distractor rung gains at least 25 points over the no-agreement control.
Useful confidence is about 0.91 while distractor confidence rises only from
0.015 to 0.055 as the set grows.

## Interpretation

This is evidence for a genuinely variable-cardinality, task-agnostic input
router rather than an N=3 special case. The agreement head was trained only on
same-frame complementary-view positives and independently rendered negatives;
the controller, encoder, and promoted bus weights are frozen. The strongest
pair rule is permutation-invariant and naturally handles additional
distractors.

The claim remains bounded: the positive relation is same-frame visual-view
agreement. Cross-modality agreement, more than five streams, and arbitrary
temporal grouping still require separate audits.

## Artifacts

- `audit_seed175101_4096.json`
- `audit_seed175201_4096.json`
- `experiments/unified_cognitive_controller/audit_amodal_pair_agreement_cardinality.py`

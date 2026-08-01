# Pair-agreement cardinality scaling through N=8 — 2026-08-01

## Breakthrough

The self-supervised pair-agreement router scales beyond the original N=3
experiment. The same frozen agreement heads and input bus were evaluated with
two useful complementary streams plus zero through six valid distractor
streams. Each event receives the strongest pair score it has with any other
event, thresholded at a fixed 0.8; no task or action information is used at
routing time.

| Cardinality | No agreement, seed 175101 | With agreement, seed 175101 | No agreement, seed 175201 | With agreement, seed 175201 |
|---:|---:|---:|---:|---:|
| N=2 | 96.38% | 96.38% | 96.38% | 96.38% |
| N=3 | 57.84% | 94.75% | 57.84% | 95.03% |
| N=4 | 55.76% | 94.37% | 55.76% | 94.57% |
| N=5 | 53.36% | 93.64% | 53.36% | 93.44% |
| N=6 | 51.87% | 92.75% | 51.87% | 92.07% |
| N=7 | 51.17% | 91.74% | 51.17% | 90.05% |
| N=8 | 50.62% | 90.35% | 50.62% | 87.96% |

Both seeds pass the cardinality gate: N=2 ≥90%, N=3 through N=8 ≥85%, and
each distractor rung gains at least 25 points over the no-agreement control.
Useful confidence is about 0.73 while distractor confidence remains below
0.033 at N=8.

## Interpretation

This is evidence for a genuinely variable-cardinality, task-agnostic input
router rather than an N=3 special case. The agreement head was trained only on
same-frame complementary-view positives and independently rendered negatives;
the controller, encoder, and promoted bus weights are frozen. The strongest
pair rule is permutation-invariant and naturally handles additional
distractors.

The claim remains bounded: the positive relation is same-frame visual-view
agreement. Cross-modality agreement, more than eight streams, and arbitrary
temporal grouping still require separate audits.

## Artifacts

- `audit_seed175101_4096.json`
- `audit_seed175201_4096.json`
- `experiments/unified_cognitive_controller/audit_amodal_pair_agreement_cardinality.py`

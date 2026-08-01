# Pair-agreement cardinality scaling — 2026-08-01

## Breakthrough

The self-supervised pair-agreement router scales beyond the original N=3
experiment. The same frozen agreement heads and input bus were evaluated with
two useful complementary streams plus zero, one, or two valid distractor
streams. Each event receives the strongest pair score it has with any other
event, thresholded at 0.4; no task or action information is used at routing
time.

| Cardinality | No agreement, seed 175101 | With agreement, seed 175101 | No agreement, seed 175201 | With agreement, seed 175201 |
|---:|---:|---:|---:|---:|
| N=2 | 96.43% | 96.43% | 96.43% | 96.43% |
| N=3 | 57.95% | 95.85% | 57.95% | 95.92% |
| N=4 | 55.27% | 93.81% | 55.27% | 93.59% |

Both seeds pass the cardinality gate: N=2 ≥90%, N=3 and N=4 ≥85%, and each
distractor rung gains at least 25 points over the no-agreement control. Useful
confidence is about 0.91 while distractor confidence is 0.015–0.037.

## Interpretation

This is evidence for a genuinely variable-cardinality, task-agnostic input
router rather than an N=3 special case. The agreement head was trained only on
same-frame complementary-view positives and independently rendered negatives;
the controller, encoder, and promoted bus weights are frozen. The strongest
pair rule is permutation-invariant and naturally handles additional
distractors.

The claim remains bounded: the positive relation is same-frame visual-view
agreement. Cross-modality agreement, more than two useful streams, and
arbitrary temporal grouping still require separate audits.

## Artifacts

- `audit_seed175101_4096.json`
- `audit_seed175201_4096.json`
- `experiments/unified_cognitive_controller/audit_amodal_pair_agreement_cardinality.py`


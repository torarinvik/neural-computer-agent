# Pair-agreement cardinality scaling through N=11 — 2026-08-01

## Breakthrough

The self-supervised pair-agreement router scales beyond the original N=3
experiment. The promoted head was trained for 256 self-supervised updates with
hidden width 64, then frozen. Two useful complementary streams plus zero
through nine valid distractor streams were evaluated. Each event receives the
strongest pair score it has with any other event, thresholded at the fixed 0.8
used by the earlier N=8 audit; no task or action information is used at
routing time.

| Cardinality | No agreement, audit A | With agreement, audit A | No agreement, audit B | With agreement, audit B |
|---:|---:|---:|---:|---:|
| N=2 | 96.47% | 96.47% | 96.19% | 96.19% |
| N=3 | 57.85% | 96.40% | 58.30% | 96.15% |
| N=4 | 55.39% | 96.13% | 55.98% | 95.76% |
| N=5 | 52.88% | 95.52% | 53.09% | 94.96% |
| N=6 | 51.25% | 94.42% | 51.67% | 94.04% |
| N=7 | 50.66% | 93.17% | 51.04% | 92.77% |
| N=8 | 50.27% | 91.73% | 50.71% | 91.23% |
| N=9 | 49.84% | 89.96% | 50.00% | 89.43% |
| N=10 | 49.75% | 88.10% | 49.96% | 87.45% |
| N=11 | 49.58% | 86.14% | 49.94% | 85.34% |

Both independent audits pass the cardinality gate: N=2 ≥90%, N=3 through
N=11 ≥85%, and each distractor rung gains at least 25 points over the
no-agreement control. Useful confidence is about 0.937 while distractor
confidence remains below 0.06 at N=11.

The first 128-update, hidden-width-32 head did not pass N=10 consistently
(81.74–82.35% on the weaker seed); a threshold sweep only reached 84.01%.
This was treated as a calibration/training limitation, not hidden by changing
the gate. A small self-supervised capacity increase produced the promoted
head. Three additional independent heads were rejected after N=10 audits,
which is why this record preserves both the population result and the
promoted checkpoint rather than claiming every initialization is equivalent.

N=12 is the current boundary: the promoted head scores 83.88% and 83.19% on
two fresh audits, below the pre-registered 85% rung gate, despite gains of
34.26 and 33.34 points over no-agreement controls. No further threshold or
architecture tuning is counted as an N=12 pass.

## Interpretation

This is evidence for a genuinely variable-cardinality, task-agnostic input
router rather than an N=3 special case. The promoted agreement head was trained
only on same-frame-view positives and independently rendered negatives; the
controller, encoder, and promoted bus weights are frozen. The strongest pair
rule is permutation-invariant and naturally handles additional distractors.

The claim remains bounded: the positive relation is same-frame visual-view
agreement. Cross-modality agreement, more than eleven streams, and arbitrary
temporal grouping still require separate audits.

## Artifacts

- `training_seed176101_h64_u256.json`
- `audit_seed176101_n11_seed181001.json`
- `audit_seed176101_n11_seed181101.json`
- `audit_seed176101_n12_seed181001.json`
- `audit_seed176101_n12_seed181101.json`
- `amodal_pair_agreement_seed176101_h64_u256.pt` (under `artifacts/checkpoints`)
- rejected-candidate reports `audit_seed176201_n10_*`,
  `audit_seed176301_n10_*`, and `audit_seed176401_n10_*`
- `experiments/archive/unified_cognitive_controller/audit_amodal_pair_agreement_cardinality.py`

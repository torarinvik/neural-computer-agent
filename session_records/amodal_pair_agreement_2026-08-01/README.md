# Self-supervised pair-agreement relevance — 2026-08-01

## Breakthrough

The amodal input bus can now learn which members of a variable-size event set
belong together, without task labels, action labels, or verifier outcomes. A
small symmetric head is trained on a task-agnostic augmentation relation:
two complementary views of the same rendered frame are positive, while a view
paired with an independently rendered frame is negative. Its pair scores are
converted into generic per-event confidence and consumed by the unchanged
confidence-weighted set bus.

On 4,096 held-out lifetimes with two useful complementary streams plus a valid
but irrelevant third stream:

| Metric | Seed 175101 | Seed 175201 |
|---|---:|---:|
| N=3 without agreement | 57.45% | 57.45% |
| N=3 with learned agreement | 89.59% | 89.25% |
| N=3 gain | +32.14pp | +31.80pp |
| N=2 retention with agreement | 98.51% | 98.55% |
| Useful-event confidence | 0.505 | 0.502 |
| Distractor confidence | 0.062 | 0.061 |

Both seeds pass the causal gate: N=2 ≥90%, N=3 ≥85%, gain ≥25 points over
the no-agreement control, and useful/distractor confidence separation ≥0.20.
The controller and promoted input bus remain frozen; only the agreement head
is trained.

## Why this matters

The earlier confidence gate required a frontend to declare low quality for an
irrelevant stream. Pair agreement removes that requirement for this setting:
the system learns relevance from the structure of the sensory augmentations
itself. This is a concrete path from fixed N=2 composition toward genuinely
variable N input routing while preserving the amodal interface.

The claim is still bounded. The positive relation is same-frame view
agreement, so arbitrary cross-modality semantic relevance and more than three
streams remain open. The next tests should vary the number of distractors,
renderers, and encoder families before promotion to a general N-to-M claim.

## Artifacts

- `training_seed175101.json`
- `training_seed175201.json`
- `audit_seed175101_4096.json`
- `audit_seed175201_4096.json`
- `experiments/archive/unified_cognitive_controller/train_amodal_pair_agreement.py`
- `experiments/archive/unified_cognitive_controller/audit_amodal_pair_agreement.py`


# Learned confidence for corrupted amodal streams — 2026-08-01

## Breakthrough

The frontend can now learn a generic event-quality confidence without task
labels, action labels, or verifier outcomes. A tiny confidence head is trained
only to predict clean-vs-corrupted latent consistency. Training includes full
frames and both complementary partial views, so valid partial modalities are
not mistaken for low quality.

The learned confidence is then attached to events and consumed by the
unchanged confidence-weighted amodal input bus. On 4,096 held-out pair-relation
lifetimes with an opaque third stream erased by 80%:

| Metric | Seed 173101 | Seed 173201 |
|---|---:|---:|
| N=2 clean with learned confidence | 99.04% | 98.77% |
| N=3 corrupted, no confidence | 81.69% | 81.69% |
| N=3 corrupted, learned confidence | 88.30% | 86.93% |
| Improvement | +6.61pp | +5.24pp |
| Mean corrupted-stream confidence (query trials) | 0.220 | 0.241 |

Both seeds pass the registered behavioral bar: N=2 ≥90%, learned N=3 ≥85%,
gain ≥5 points over the no-confidence control, and mean third-stream confidence
≤0.35. The controller, visual encoder, and promoted input bus weights remain
unchanged; only the confidence head is trained.

## What remains open

This is a corruption/missing-evidence result, not arbitrary distractor
relevance. A valid but semantically irrelevant third stream is not necessarily
low quality, and the earlier confidence-routing audit still requires the
frontend to provide a low confidence for that case. The next frontier is a
task-agnostic relevance or agreement estimator for valid competing streams,
with the same N=2 retention and shuffled-stream controls.

## Training and audit method

For each synthetic lifetime, the frozen vision encoder emits a clean latent and
a randomly erased latent. The confidence head regresses the scalar
`exp(-scale * latent_mse)`. The training distribution includes full images and
the two complementary partial views. No semantic target is constructed.

Artifacts:

- `training_seed173101.json`
- `training_seed173201.json`
- `audit_seed173101_4096.json`
- `audit_seed173201_4096.json`
- `experiments/unified_cognitive_controller/train_amodal_confidence_estimator.py`
- `experiments/unified_cognitive_controller/audit_amodal_learned_confidence.py`


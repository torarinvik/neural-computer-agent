# Confidence-gated N=3 input composition — 2026-08-01

## Breakthrough

The promoted, frozen amodal input bus already supports a useful missing- or
irrelevant-stream policy when an encoder supplies its generic confidence. On
held-out pair-relation lifetimes, two complementary streams score 96.58%.
Adding an opaque third stream with full confidence drops behavior to 58.42%.
The same third stream with low confidence is suppressed by the existing
confidence-weighted set bus, recovering 91.12% at confidence 0.1 and 96.40%
at confidence 0.01. Confidence 0.0 exactly recovers the two-stream score.

| Third-stream confidence | N=3 accuracy |
|---:|---:|
| 1.00 | 58.42% |
| 0.50 | 70.15% |
| 0.10 | 91.12% |
| 0.01 | 96.40% |
| 0.00 | 96.58% |

The pre-registered audit passed: N=2 remains at least 90%; full-confidence N=3
is at or below 65%; low-confidence N=3 is at least 90%, within two percentage
points of N=2, and improves by at least 25 points over the no-confidence
control. The controller, encoder, and bus weights are unchanged.

## What this proves — and what it does not

This qualifies the confidence transport and routing mechanism as a genuine
amodal N-to-3 capability. It uses no task labels, action labels, semantic
identities, or optimizer updates. Confidence is a generic frontend quality
signal already present in the event schema; the controller never sees the
reason for the confidence value.

The audit supplies confidence values directly, so it does **not** yet prove
that a learned encoder can estimate confidence. That is the next frontier:
learn confidence from task-agnostic clean/corrupted consistency or predictive
uncertainty, then repeat this exact causal audit with the confidence predictor
frozen and independently evaluated.

## Rejected learning-only attempts

Three tiny scalar-reward N=3 runs and two latent-consistency runs improved the
N=3 score only a few points and either damaged N=2 retention or failed the
irrelevant-stream control. They remain recorded as bounded negative evidence:
the generic bus cannot reliably infer an irrelevant stream from content alone
at this sample scale. Confidence metadata is therefore the high-ROI bridge,
not an excuse to promote those weights.

## Artifact

- `confidence_audit_4096.json`
- `rejected_scalar_n3_8update.json`
- `rejected_scalar_n3_frozen_rehearsal.json`
- `rejected_scalar_n3_rehearsal.json`
- `rejected_consistency_8update.json`
- `rejected_consistency_64update.json`
- `experiments/unified_cognitive_controller/audit_amodal_n3_confidence.py`


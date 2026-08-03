# Amodal input-noise adaptation boundary — 2026-08-01

## Result

The promoted frozen complementary input bus remains the best current input
adapter. Two small outcome-only fine-tuning attempts were deliberately tested
against deterministic pixel erasure and rejected: both retained clean
behavior, but both became less robust as corruption increased. No adapted
checkpoint was promoted.

This is a negative localization result, not evidence that the amodal bus is
fragile. The baseline bus already handles partial complementary evidence and
retains useful behavior under substantial corruption. The tested update rule
only teaches the bus from scalar success on the clean task; it does not expose
which pixels were corrupted or provide a corruption-invariant target. It can
therefore improve the clean shortcut while damaging the margin needed for
missing evidence.

## Audits

The audits use the same frozen controller and held-out pair-relation lifetimes.
Only the input bus differs. The second partial stream is independently erased
at the listed fraction; the correct action remains verifier-side.

| Erasure | Frozen bus | 8-update adapted | 24-update adapted |
|---:|---:|---:|---:|
| 0.0 | 0.9625 | 0.9645 | 0.9693 |
| 0.2 | 0.9378 | 0.9193 | 0.9410 |
| 0.4 | 0.8948 | 0.8512 | 0.8778 |
| 0.6 | 0.8176 | 0.7500 | 0.7809 |
| 0.8 | 0.6980 | 0.6385 | 0.6646 |

The 8-update run used 1,536 verifier bits and 40% training erasure. The
24-update run used 4,608 verifier bits and 60% training erasure. Both passed
the trainer's clean invariants but failed the independent noise audit's
required noisy gains. A separate small confidence-weighting diagnostic also
decreased performance monotonically, so it was not promoted either.

## Interpretation and next boundary

The current evidence rules out blind scalar-reward adaptation as a high-ROI
robustness fix. The next useful experiment is a task-agnostic corruption-aware
frontend or learned uncertainty/denoising mechanism with an explicit
clean-retention and corruption-curve gate. It must be tested against shuffled
corruptions and held-out renderers before any bus weights are promoted.

The timestamp-aware asynchronous transport breakthrough remains valid and
separate: timestamp-sorted out-of-order and bounded-jitter streams reproduced
synchronous behavior at 96.36%. This noise study closes the next boundary
without changing that result or the promoted frozen bus.

## Artifacts

- `audit_baseline_vs_adapted_8update_4096.json`
- `audit_baseline_vs_adapted_24update_4096.json`
- `training_outcome_only_8update.json`
- `training_outcome_only_24update.json`
- `experiments/archive/unified_cognitive_controller/audit_amodal_input_noise.py`


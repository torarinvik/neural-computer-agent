# Calibrated replay-free streaming admission

This pressure test connects the existing frozen transition-evidence evaluator
and external contextual calibrator to the streaming candidate router. Two
nonlinear transition streams are acquired with one-pass random-feature
statistics while the controller remains frozen. A low-error but corrupted
stream is then rejected by learned evidence calibration even though its raw
prediction error is below the router's continuation tolerance.

The calibrator is external, context-addressed, independently persisted, and
trained from scalar verifier outcomes. No controller parameter or candidate
raw row is updated by the calibration path. This is a bounded learned
reliability gate, not general continual learning or learned delay handling.

```text
PYTHONPATH=src .venv/bin/python \
  experiments/external_calibrated_streaming_admission/train.py \
  --seed 2001 \
  --report-out /tmp/external-calibrated-streaming-admission.json
```

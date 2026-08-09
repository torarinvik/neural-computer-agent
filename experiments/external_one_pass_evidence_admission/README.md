# One-pass evidence admission

This audit replaces replay-heavy evidence-evaluator pretraining with an
external error-bin sufficient-statistics model. Scalar verifier outcomes update
only positive/negative counts; no evaluator optimizer or replay is used. The
learned gate is connected to the streaming nonlinear candidate router, which
acquires sequential regimes and still uses an explicit warm-up before
reliability can veto a provisional candidate's continuation.

```text
PYTHONPATH=src:. .venv/bin/python \
  experiments/external_one_pass_evidence_admission/train.py \
  --seed 2101 \
  --report-out /tmp/external-one-pass-evidence-admission.json
```

This is a bounded replay-free reliability primitive, not general continual
learning, unrestricted nonlinear computation, or learned delay handling.

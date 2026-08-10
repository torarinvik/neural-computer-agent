# Goal representation drift gate

This pressure test verifies the learned-alignment promotion boundary. A
partially observed affine replacement frontend must pass held-out alignment
verification and preserve planning. A genuinely nonlinear replacement is
fit with the same replay-free linear adapter, but its held-out error must cause
the candidate to be rejected before it can serve the old goal memory.

The old goal verifier statistics remain frozen and no verifier outcomes are
replayed. This is a safety and evidence-quality rung, not a claim that the
linear adapter solves nonlinear representation migration.

Run one seed with:

```bash
PYTHONPATH=src:. .venv/bin/python \
  experiments/external_goal_representation_drift_gate/train.py \
  --seed 84401 \
  --report-out /tmp/external-goal-representation-drift-gate-84401.json
```

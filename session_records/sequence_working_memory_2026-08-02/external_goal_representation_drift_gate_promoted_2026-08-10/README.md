# Goal representation drift gate — promoted

This four-seed audit tests whether a learned frontend alignment is safe to
promote. A partially observed affine replacement frontend is aligned into the
frozen goal-memory space and must pass held-out verification before planning.
A genuinely nonlinear replacement is fit with the same replay-free linear
adapter, but is rejected when its held-out behavior exceeds tolerance.

Across seeds `84401`, `84402`, `84403`, and `84404`, partial-affine planning
reached `120/120` trials on every seed. Its maximum held-out alignment MSE was
between `5.18e-6` and `9.09e-6`. The nonlinear candidate's held-out MSE was
between `0.299` and `0.304` and was rejected on every seed. Rejected
candidates were never served to the planner or live memory.

The old goal verifier memory remained unchanged, verifier replay was zero, the
controller and factual model stayed frozen, and the accepted adapter restored
exactly. This promotes evidence-gated acceptance/rejection, not nonlinear
alignment itself or general continual learning. The next step is to learn a
nonlinear external alignment basis or retain the new frontend in quarantine
until enough verified evidence supports one.

Reproduce one seed with:

```bash
PYTHONPATH=src:. .venv/bin/python \
  experiments/external_goal_representation_drift_gate/train.py \
  --seed 84401 \
  --report-out /tmp/external-goal-representation-drift-gate-84401.json
```

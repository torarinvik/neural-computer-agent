# Replay-free nonlinear feature memory

This pressure test evaluates the fixed nonlinear feature family introduced
after the one-pass MLP rejection. The memory consumes an opaque nonlinear
transition stream once, stores no raw rows, and predicts held-out transitions
through persisted sufficient statistics.

```text
.venv/bin/python experiments/external_random_feature_one_pass/train.py \
  --seed 1401 \
  --report-out /tmp/external-random-feature-one-pass.json
```

This is a promoted bounded mechanism, not general continual learning. The
feature basis is fixed during this audit, and the report does not claim
unrestricted computation or robustness to distribution shift.

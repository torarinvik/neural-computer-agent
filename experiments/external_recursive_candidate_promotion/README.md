# Recursive candidate promotion

This pressure test verifies the canonical copy-on-write promotion transaction:
one-step held-out fit, recursive held-out rollout, and retention must all pass
before a candidate or capacity growth reaches live external memory.

```text
PYTHONPATH=src:. uv run python \
  experiments/external_recursive_candidate_promotion/train.py \
  --seed 84001 \
  --report-out /tmp/external-recursive-candidate-promotion.json
```

The result is intentionally bounded to a tiny affine dynamics family. Its
purpose is to protect the promotion invariant before applying it to noisy,
partial nonlinear streams.

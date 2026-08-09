# Replay-free nonlinear slot retention

This audit trains four disjoint nonlinear transition families into isolated
external sufficient-statistics slots. Each slot consumes its 64-row training
stream once; after all later families are acquired, every earlier slot is
revisited on held-out evidence and its digest is compared with its post-learn
digest. The promoted rerun uses ridge `1e-4`; the rejected `1e-5` configuration
is archived separately.

```text
.venv/bin/python experiments/external_random_feature_retention/train.py \
  --seed 1501 \
  --report-out /tmp/external-random-feature-retention.json
```

The context keys are verifier-supplied one-hot fixtures, so this promotes a
bounded retention result, not learned unrestricted routing or general
continual learning.

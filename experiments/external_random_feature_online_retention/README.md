# Online replay-free nonlinear retention with learned context formation

This pressure test removes supplied context keys from the routing path. Four
disjoint nonlinear transition streams arrive one row at a time; the learned
context encoder forms opaque candidate keys, the router separates committed
continuation from provisional accumulation, and the verifier promotes each
candidate only if prior slots retain their held-out floors.

```text
.venv/bin/python experiments/external_random_feature_online_retention/train.py \
  --seed 1601 \
  --report-out /tmp/external-random-feature-online-retention.json
```

The fixture is still bounded and the context encoder is not trained in this
audit. It promotes online routing plus replay-free external retention, not
general continual learning.

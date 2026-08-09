# Learned context-address admission

This pressure test removes supplied context labels from the disjoint-dynamics
rung. A memory-side resolver receives only opaque state/intention/next-state
transition bundles. It reuses an existing address only when the append-only
factual store explains every row; otherwise it allocates a fresh opaque handle.

The fixture presents three unique dynamics regimes and one duplicate regime.
It tests address reuse, reversal retention, shuffled-context, corrupted-memory,
fresh-memory, and persistence controls while the controller remains frozen.

```text
.venv/bin/python experiments/external_context_address_transfer/train.py \
  --seed 69501 \
  --report-out /tmp/external-context-address-transfer.json
```

This is still bounded nonparametric continual memory. It does not establish
learned context formation from raw modalities, compression, or extrapolation
beyond stored transition facts.

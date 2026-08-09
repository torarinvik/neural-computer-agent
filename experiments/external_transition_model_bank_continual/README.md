# External transition-model bank continual learning

This pressure test follows the exported-session result that a policy-free
learned model is the promising route to compounding acquisition cost. It
trains one opaque transition model on a source dynamics regime, appends a
target slot initialized from that model, and adapts only the target slot.
Behavior is derived by `ExternalModelBasedPlanner` search; no policy is
stored in the controller.

The target phase does not replay source observations. It does reuse its
current target batch for optimization, and that cost is reported separately
from old-source replay. Source retention, wrong-context, corruption, fresh
model, persistence, and frozen-controller controls are included.

```text
.venv/bin/python experiments/external_transition_model_bank_continual/train.py \
  --seed 69811 \
  --report-out /tmp/model-bank-continual.json
```

This is a bounded external model-bank result. Context vectors are supplied,
the bank grows append-only, and it does not yet learn context discovery,
compress slots, or demonstrate a general downward acquisition-cost curve.

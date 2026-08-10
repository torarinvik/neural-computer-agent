# Disjoint-dynamics policy-free compounding

This is the next pressure test after the nested model-compounding result. It
keeps the controller frozen, learns two source transition regimes, and then
acquires two genuinely disjoint target dynamics sequentially. Warm target
models inherit only the immediately preceding factual model; matched fresh
models start from new weights. Behavior is derived at inference time by
opaque model search, with no task policy stored in the controller or bank.

Before each target is appended, the bank creates isolated transfer and fresh
challengers and gives both the same four-update factual prefix. The lower-loss
candidate is selected, while the source slot remains byte-stable. The fresh
challenger is cloned from one caller-owned baseline, and the matched fresh
control trains from that exact unprobed state; this removes initialization luck
from the transfer comparison. This copy-on-write gate is the mechanism under
test; it is not allowed to silently turn a negative-transfer seed into a pass.

The audit charges matched source acquisition, stops only when both factual
loss and planner mastery pass, measures current-target reuse separately from
old-regime replay, and checks every prior slot after every target. It is a
replication/extension of the exported session's F69 mechanism, not a claim of
general continual learning.

The current source also measures a verifier-only random-intention floor: 128
trials per target, 768 target trials per seed. This keeps the planner result
separate from the no-agent baseline; the floor must remain below the mastery
threshold for promotion.

Run one seed with:

```bash
PYTHONPATH=src:. .venv/bin/python \
  experiments/external_transition_model_disjoint_compounding/train.py \
  --seed 70411 --report-out /tmp/external-disjoint-compounding-70411.json
```

# Active factual-model disambiguation probe

This audit tests the exported session's probe-addressing mechanism. Two
opaque external factual models agree on the current observation and differ
only in the consequence of one available intention. The planner selects the
intention with maximal predicted disagreement; the observed consequence then
routes the hidden model. A uniform random-intention control and random tie
break establish the causal floor.

The controller is frozen, transition acquisition is one-pass, queries do not
mutate the model bank, and persistence is checked. This is a narrow causal
probe boundary, not a claim of multimodal probe learning or general continual
learning.

```text
uv run python experiments/external_model_disambiguation_probe/train.py \
  --seed 83001 \
  --report-out /tmp/external-model-disambiguation-probe.json
```

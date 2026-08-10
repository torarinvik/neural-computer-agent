# Concurrent stream robustness

This pressure test extends the shared factual-bank boundary with the next
required lifecycle: one stream can be delayed or missing, another can emit
contradictory evidence, and a bounded bank can evict and replace a drifted
stream only after verifying retention of the other stream.

The stream keys are opaque caller-owned binding tokens. They are not task
labels, so this tests safe transport/factual-memory behavior rather than
learned identity formation.

Run it from the repository root:

```bash
.venv/bin/python \
  experiments/external_transition_model_multistream_robustness/train.py \
  --seed 2201
```

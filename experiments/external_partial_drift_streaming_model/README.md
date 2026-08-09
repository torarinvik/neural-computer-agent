# Replay-free partial-evidence and gradual-drift model routing

This pressure test targets the next bottleneck after model compounding. A
factual transition regime is presented through only `8` of `12` available
rows, and the external router consumes each presented row once through affine
sufficient statistics. Two gradually drifted versions are then learned as
new copy-on-write slots. The original slot is never updated or replayed.

After each drift version is promoted, the stream returns to the original
regime. The router must select the old slot, preserve its planner-level
behavior, and avoid minting a duplicate. Candidate promotion is held-out and
retention verified; a shuffled/corrupted stream is deliberately rejected
without changing the committed bank.

This is a bounded replay-free factual-memory result. It does not establish
learned multimodal context formation, unrestricted memory growth, or general
continual learning.

Run one seed with:

```text
PYTHONPATH=src:. .venv/bin/python \
  experiments/external_partial_drift_streaming_model/train.py \
  --seed 81001 --report-out /tmp/external-partial-drift-81001.json
```

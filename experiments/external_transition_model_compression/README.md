# External transition-model storage compression

This pressure test evaluates compressed external model-state candidates against
held-out factual transition loss. The controller remains frozen; compression
does not add a reasoning branch. Float16 and int8 storage candidates must pass
the same source/target retention probe and round-trip exactly through the
versioned compressed payload boundary. An int4 candidate is included as a
stricter control and is rejected when its held-out loss drift exceeds tolerance.

The result concerns checkpoint/external-memory bytes, not live float32 runtime
parameter dtype. A caller must promote the compressed artifact only after the
probe passes.

```text
.venv/bin/python experiments/external_transition_model_compression/train.py \
  --seed 70211 \
  --report-out /tmp/transition-model-compression.json
```

# Replay-free streaming-statistics candidates

This pressure test closes a boundary leak in online factual-memory admission.
The provisional router consumes each transition exactly once through a
random-feature sufficient-statistics model. It does not retain the raw
provisional rows, does not replay committed rows, and promotes only after a
held-out factual probe and an old-slot retention probe pass.

The matched shuffled-next-state control receives the same stream shape and
budget but cannot predict the held-out factual transitions. The controller is
frozen throughout; behavior is still derived from the external model path.

```text
.venv/bin/python experiments/external_streaming_statistics_candidate/train.py \
  --seed 1801 \
  --report-out /tmp/external-streaming-statistics-candidate.json
```

This is a bounded one-pass sufficient-statistics result, not general
continual learning. The random feature basis is fixed and finite; broader
nonlinear model classes still require either replay or a different streaming
learner.

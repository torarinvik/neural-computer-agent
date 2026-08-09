# Interleaved streaming factual candidates

This pressure test presents two novel dynamics streams in alternating partial
windows before either candidate is promoted. The external router must keep
their provisional sufficient statistics isolated, choose a model family only
through held-out factual verification, and refuse a third stream when capacity
is full. The controller is frozen and no provisional raw rows are retained.

```text
.venv/bin/python experiments/external_interleaved_streaming_candidates/train.py \
  --seed 1901 \
  --report-out /tmp/external-interleaved-streaming-candidates.json
```

The fixture uses two disjoint affine transition functions and offers affine
and fixed random-feature sufficient-statistics candidates. This is an
interleaved replay-free boundary result, not arbitrary nonlinear continual
learning or unrestricted computation.

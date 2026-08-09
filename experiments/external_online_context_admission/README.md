# Online partial-evidence context admission

This pressure test exercises the next continual-learning bottleneck after
exact-bundle address admission. Three opaque streams are interleaved one
transition at a time. The resolver keeps the first two observations of each
stream provisional and unwritten; after the third it admits and commits a new
opaque address. A later reversal on an already-bound stream requires two
contradictions before receiving a new address, preserving the old facts.

```text
.venv/bin/python experiments/external_online_context_admission/train.py \
  --seed 69601 \
  --report-out /tmp/external-online-context-admission.json
```

This remains a bounded memory-side pressure test. It does not yet learn from
raw modality streams, resolve arbitrary partial observations, or compress
unbounded history.

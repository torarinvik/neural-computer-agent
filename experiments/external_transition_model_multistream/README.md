# Interleaved external transition-model streams

This is a sub-minute structural pressure test for the architecture learned
from the exported games session: one fixed controller boundary, one shared
factual model bank, and independently addressable stream-local learning state.

Three opaque streams are interleaved. Each stream must accumulate its own
evidence window, stage its own copy-on-write candidate, and promote without
altering the other streams' candidates. After promotion, the streams are
revisited in an interleaved order, persisted, restored, and routed again.

The stream key is an opaque binding token supplied by the event-binding layer;
it is not a task label or a learned-capability claim. The test therefore
promotes only the transport and factual-memory invariant. It does not claim
general identity formation, arbitrary program induction, unrestricted memory
growth, or general continual learning.

Run it from the repository root:

```bash
.venv/bin/python experiments/external_transition_model_multistream/train.py \
  --seed 1901
```

# Dynamic-regime factual versioning

This pressure test verifies a missing lifecycle property in online continual
memory: when one opaque stream changes from regime A to regime B, B receives a
new factual address while A remains intact; when the stream returns to A or B,
the resolver reactivates the retained version instead of allocating A-v2 or
overwriting either fact set.

The controller is frozen. The test uses one shared opaque stream binding,
single verified transitions, no optimizer updates, no replay, persistence,
and corrupted/fresh memory controls. It is a bounded factual-memory result,
not evidence for unrestricted continual learning or learned regime discovery
from raw modalities.

```text
uv run python experiments/external_online_context_versioning/train.py \
  --seed 96001 \
  --report-out /tmp/external-online-context-versioning.json
```

# Factored external-memory lifecycle pressure test

This pressure test exercises the canonical factored router across more logical
regimes than its initial capacity. A frozen zero base is paired with a
replay-free nonlinear random-feature residual bank. Two regimes are promoted,
capacity is then grown to four under retention verification, two more regimes
are promoted, one middle logical slot is evicted under a retention probe, and
a fifth regime is admitted using the surviving stable addresses.

The test also selects and round-trips a storage-compressed residual-bank
checkpoint. All promotions, growth, eviction, routing, and compression checks
use independent held-out observations. No controller, base, or context encoder
weights are updated, and no old-regime rows are replayed during new-regime
adaptation.

This is a bounded memory-lifecycle result. It does not establish automatic
context discovery in open-ended environments, unrestricted capacity, learned
compression policy, or general continual learning.

Run one seed with:

```bash
PYTHONPATH=src uv run python experiments/external_factored_memory_lifecycle/train.py \
  --seed 81031 --report-out /tmp/external-factored-memory-lifecycle.json
```

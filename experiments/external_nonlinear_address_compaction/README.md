# External nonlinear address compaction

This pressure test composes the promoted learned-address nonlinear stream with
retention-verified external-memory lifecycle operations:

- grow capacity from four to six and then seven slots;
- create a copy-on-write equivalent factual slot;
- consolidate only the equivalent slots while preserving both opaque logical
  addresses;
- select a statistics-aware float16 codec only after a held-out retention
  probe, while rejecting unsafe legacy codecs;
- reject corrupted evidence without changing the committed bank.

The controller is frozen. Source examples are not replayed during target
acquisition or compaction. Run from the repository root with `PYTHONPATH=.`:

```bash
PYTHONPATH=. .venv/bin/python \
  experiments/external_nonlinear_address_compaction/train.py \
  --seed 82301 --report-out /tmp/external-address-compaction-82301.json
```

The random-feature sufficient-statistics family rejects the legacy per-tensor
float16/int8 codecs when their quantization changes held-out factual
predictions. The new `float16_stats` codec preserves the immutable basis and
ill-conditioned normal matrix, stores the solved predictor in float16, and
reconstructs the target statistics on restore. It is promoted only when it
passes the same verifier. Compression is never accepted just to reduce bytes.

This establishes a bounded lifecycle contract, not semantic model merging,
unrestricted memory growth, or general continual learning.

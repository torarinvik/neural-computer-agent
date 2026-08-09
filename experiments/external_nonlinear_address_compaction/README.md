# External nonlinear address compaction

This pressure test composes the promoted learned-address nonlinear stream with
retention-verified external-memory lifecycle operations:

- grow capacity from four to six and then seven slots;
- create a copy-on-write equivalent factual slot;
- consolidate only the equivalent slots while preserving both opaque logical
  addresses;
- test storage codecs and reject them when sufficient-statistics quantization
  causes held-out drift;
- reject corrupted evidence without changing the committed bank.

The controller is frozen. Source examples are not replayed during target
acquisition or compaction. Run from the repository root with `PYTHONPATH=.`:

```bash
PYTHONPATH=. .venv/bin/python \
  experiments/external_nonlinear_address_compaction/train.py \
  --seed 82301 --report-out /tmp/external-address-compaction-82301.json
```

The random-feature sufficient-statistics family intentionally rejects the
current float16/int8 codecs when their quantization changes held-out factual
predictions. This is a promoted safety result: compression is never accepted
just to reduce bytes. A statistics-aware codec is a separate follow-up.

This establishes a bounded lifecycle contract, not semantic model merging,
unrestricted memory growth, or general continual learning.

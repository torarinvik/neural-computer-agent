# Partial and ambiguous open-world memory

This experiment combines online opaque address formation, nonlinear factual
memory, partial evidence, contradiction quarantine, concurrent copy-on-write
candidates, and recursive rollout-gated promotion.

```text
PYTHONPATH=src:. uv run python \
  experiments/external_partial_ambiguous_open_world/train.py \
  --seed 82501 \
  --report-out /tmp/external-partial-ambiguous.json
```

The promoted result is intentionally bounded. It tests the memory boundary
under partial and ambiguous evidence, not general continual learning or
unrestricted external-memory growth.

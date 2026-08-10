# Verifier-gated masked identity-memory growth

This experiment pressure-tests the next boundary after masked-prototype
replacement: external identity memory may increase its per-slot prototype
budget only after a copy-on-write candidate passes a retention probe.

The controller, transition model, verifier statistics, and alignment adapters
remain frozen. Two opaque alignment slots are admitted once. The identity
memory starts with one prototype per slot, grows to three through the new
verifier-gated API, and then learns two distinct partial observations under a
shared evidence mask for one slot without replaying the original observations
or replacing them. Cross-mask transfer remains a separate pressure test.

Run one seed with:

```bash
PYTHONPATH=. .venv/bin/python \
  experiments/external_goal_alignment_masked_growth/train.py \
  --seed 85201 \
  --report-out /tmp/masked-growth-85201.json
```

This is a bounded external-memory-growth result. It does not establish an
autonomous retention policy, unbounded memory, semantic open-world identity,
or general continual learning.

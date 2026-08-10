# Verifier-gated executable-memory lifecycle

This pressure test treats learned executable artifacts as files owned by an
external memory system. One shared register interpreter and the amodal
controller remain frozen while the memory performs four audited operations:

- reject eviction of a protected file;
- evict an unprotected file after held-out retention verification;
- consolidate two files only after held-out execution equivalence verification;
- compress durable storage only after decompression and retention verification.

Logical file IDs survive physical compaction. Failed probes, corrupted
payloads, and mutating probes must leave the live bank unchanged. The
controller never receives a file ID, task label, verifier target, or raw
verification row.

Run a seed with:

```bash
PYTHONPATH=src:. .venv/bin/python \
  experiments/external_program_memory_lifecycle/train.py \
  --seed 24001 \
  --report-out /tmp/external-program-memory-lifecycle-24001.json
```

This promotes bounded, verifier-gated lifecycle management of an external
executable file bank. It does not prove unrestricted memory growth, arbitrary
program synthesis, or general continual learning.

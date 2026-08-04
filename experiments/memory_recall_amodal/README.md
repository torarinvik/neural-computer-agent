# Outcome-only memory recall

This is the smallest memory-dependent rung for the canonical runtime. The
verifier hides a binary probe target and returns only its scalar outcome. The
controller then receives that opaque action/outcome feedback, writes a learned
memory value, resets its recurrent state, and must reproduce the outcome with
no sensory evidence.

The diagnostic uses a one-row store with a `0.5` write threshold. The
write-strength path is differentiable through
`MemoryBackend.differentiable_transaction()`: the persistent row is still
detached and serialized normally, while the current episode's write/read path
remains differentiable. The ordinary seeds commit at rates `1.0`, `0.6563`,
and `1.0`; this is evidence that the gate can affect writes, but not a
population qualification of learned skipping or utility-based retention. This
is a narrow scalar-outcome memory result, not a claim of general episodic
memory or multimodal transfer.

Start with:

```bash
PYTHONPATH=src .venv/bin/python -m experiments.memory_recall_amodal.train \
  --steps 256 --seed 17 --report-out /tmp/memory-recall.json
```

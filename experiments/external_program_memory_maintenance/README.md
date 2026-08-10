# Learned maintenance over executable external memory

This is the first audit that connects the generic learned maintenance policy
to the executable-file store itself. A frozen shared register interpreter
executes opaque files while an external policy chooses among legal `grow`,
`share`, `compress`, `evict`, and `defer` operations. The policy receives only
generic storage telemetry and an action mask. Actual changes still require the
file bank's verifier-gated copy-on-write transactions.

The audit uses real executable artifacts and held-out register execution for
retention/equivalence. It also checks corrupted compression, exact payload
reload, canonical runtime traversal, a frozen interpreter/controller, and zero
replay. It promotes policy integration with this file backend; it does not
establish autonomous verifier design, unrestricted growth, arbitrary program
synthesis, or general continual learning.

Run a seed with:

```bash
PYTHONPATH=src:. .venv/bin/python \
  experiments/external_program_memory_maintenance/train.py \
  --seed 25001 \
  --report-out /tmp/external-program-memory-maintenance-25001.json
```

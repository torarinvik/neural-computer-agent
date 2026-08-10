# Verifier-gated intention admission

This pressure test extends the external intention repertoire with a
copy-on-write admission transaction. A novel opaque intention is staged,
tested against independent held-out factual evidence, and committed only when
all retained repertoire vectors remain unchanged. A rejected candidate is a
complete no-op.

The promoted run uses a diagonal intention that is absent from the initial
cardinal repertoire. A three-step goal requiring three diagonal intentions is
unreachable before admission and mastered afterward without updating the
controller or factual model.

Run one seed with:

```bash
PYTHONPATH=. .venv/bin/python experiments/policy_free_intention_admission/train.py \
  --seed 85201 \
  --report-out /tmp/policy-free-intention-admission-85201.json
```

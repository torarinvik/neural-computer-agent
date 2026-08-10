# Verifier-gated intention admission

This pressure test extends the external intention repertoire with a
copy-on-write admission transaction. A novel opaque intention is proposed by
composing retained opaque entries, staged, tested against independent held-out
factual evidence, and committed only when all retained repertoire vectors
remain unchanged. A rejected candidate is a complete no-op.

The promoted run uses the composition explorer's mean of the first two
cardinal entries (`[0.5, 0.5]`), which is absent from the initial repertoire. A
three-step goal requiring three diagonal intentions is unreachable before
admission and mastered afterward without updating the controller or factual
model. A second proposed difference candidate (`[1.0, -1.0]`) is rejected by
the held-out verifier and leaves the live repertoire unchanged.

Run one seed with:

```bash
PYTHONPATH=. .venv/bin/python experiments/policy_free_intention_admission/train.py \
  --seed 85201 \
  --report-out /tmp/policy-free-intention-admission-85201.json
```

# Drift, missing evidence, and caller-free identity anchors

This pressure test composes the delayed identity boundary with three harder
conditions: opaque identity signatures drift gradually, drift reverses several
times, and some queries arrive with masked learned evidence. Query order also
reverses so runtime routing cannot depend on stream position.

The bank selects the proposed identity slot from the signature itself. A
verifier accepts or rejects that proposal without supplying a frontend or slot
ID. Full accepted anchors may update identity prototypes and resolve deferred
evidence; partial anchors route but are not stored as incomplete prototypes.

Run one seed with:

```text
PYTHONPATH=.:src .venv/bin/python experiments/external_goal_alignment_drift_missing_reversal/train.py \
  --seed 85001 --report-out /tmp/external-goal-alignment-drift-missing-reversal.json
```

The claim is intentionally bounded: replay-free, verifier-gated identity
retention under gradual drift, reversible drift, and partial evidence. It does
not establish open-world identity discovery, autonomous verifier design,
unrestricted growth, or general continual learning.

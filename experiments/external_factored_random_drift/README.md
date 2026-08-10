# Randomized partial evidence with factual drift

This experiment composes two already-promoted external-memory mechanisms. Four
opaque nonlinear regimes are first admitted through randomized seven-row
windows. Each retained regime then receives a disjoint four-row drift update
from an independently randomized eight-row stream, with the other four rows
held out.

The controller, shared base, and context encoder remain frozen. Drift is
copy-on-write and is committed only when its held-out predictions pass and the
previous regime's held-out behavior is retained. The sparse factual index is
persisted and tested only as a proposal accelerator; it does not replace the
factual gates.

Across five seeds, all initial regimes and drift versions promoted, randomized
partial reads routed the correct slots, mixed evidence remained ambiguous,
state round-tripped exactly, and old behavior remained within tolerance. No
old-regime rows were replayed during drift.

This promotes a bounded composition result: replay-free randomized partial
evidence plus gradual factual drift. It does not establish learned semantic
identity, arbitrary open-world version formation, unrestricted memory growth,
or general continual learning.

Run one seed with:

```bash
PYTHONPATH=src uv run python experiments/external_factored_random_drift/train.py \
  --seed 84041 --report-out /tmp/external-factored-random-drift.json
```

Promoted reports and accounting are archived in
`session_records/sequence_working_memory_2026-08-02/external_factored_random_drift_promoted_2026-08-10/`.

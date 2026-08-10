# Caller-free learned intention-cell routing — 2026-08-10

This audit tests the next external-memory seam after independent cells and
negative-transfer rollback. `ExternalOutcomeIntentionRouter` selects an
external intention cell from opaque controller context and exploration. The
runtime receives only the selected opaque intention; the trainer never passes a
cell index to the runtime. Delayed scalar verifier feedback updates the
selected content and route propensity without replay or controller updates.
Routing happens before content generation, so each single-context step
materializes one candidate even when the external bank contains three cells.

| seed | source updates / score | warm successor updates / score | matched fresh successor updates / score | reversal score | promoted |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 85301 | 24 / 0.978 | 7 / 0.951 | 46 / 0.954 | 0.978 | yes |
| 85302 | 38 / 0.973 | 16 / 0.965 | 41 / 0.956 | 0.934 | yes |

Both seeds automatically select the appended successor cell, reject an
inherited reversal by a held-out negative-transfer probe, learn a fresh
reversal under 20% outcome noise, preserve protected cell content, and pass
reward-shuffled, action-shuffled, missing-evidence, corruption, persistence,
frozen-controller, sparse-materialization, and zero-replay gates.
The matched fresh-cell transfer ratios are 6.57x and 2.56x fewer warm updates.

The claim boundary is bounded caller-free routing over growing external
intention cells. It is not unrestricted continual learning, compression,
arbitrary new computation, or Brain Workshop mastery. The next task is to
repeat the transfer result over a longer stable-prefix ledger while reducing
route/verifier cost and retaining all mastered cells.

Reproduce with:

```bash
PYTHONPATH=src:. .venv/bin/python \
  experiments/policy_free_intention_routing/train.py \
  --seed 85301 \
  --report-out /tmp/policy-free-intention-routing-85301.json
```

Reports:

- `report_seed_85301.json`
- `report_seed_85302.json`
- `sample_efficiency_ledger.json`
- `SHA256SUMS`

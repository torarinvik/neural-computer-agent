# Caller-free learned intention-cell routing — 2026-08-10

This audit tests the next external-memory seam after independent cells and
negative-transfer rollback. `ExternalOutcomeIntentionRouter` selects an
external intention cell from opaque controller context and exploration. The
runtime receives only the selected opaque intention; the trainer never passes a
cell index to the runtime. Delayed scalar verifier feedback updates the
selected content and route propensity without replay or controller updates.

| seed | source updates / score | successor updates / score | fresh successor updates / score | reversal score | promoted |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 85301 | 31 / 0.952 | 35 / 0.985 | 32 / 0.973 | 0.919 | yes |
| 85302 | 35 / 0.957 | 20 / 0.951 | 42 / 0.965 | 0.905 | yes |

Both seeds automatically select the appended successor cell, reject an
inherited reversal by a held-out negative-transfer probe, learn a fresh
reversal under 20% outcome noise, preserve protected cell content, and pass
reward-shuffled, action-shuffled, missing-evidence, corruption, persistence,
frozen-controller, and zero-replay gates.

The claim boundary is bounded caller-free routing over growing external
intention cells. It is not unrestricted continual learning, compression,
arbitrary new computation, or Brain Workshop mastery. The earlier warm-transfer
speedup is not promoted here: route exploration and competing-cell trials make
successor updates comparable to the matched fresh control. The next task is to
reduce route/verifier cost with a stable-prefix transfer ledger while retaining
all mastered cells.

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

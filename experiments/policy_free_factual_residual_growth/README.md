# Policy-free factual residual growth

This pressure test promotes the next architectural seam after intention-cell
retention: a frozen shared factual transition model plus a context-addressed
external residual. The residual uses random-feature statistics, so each fresh
transition row is consumed once without replay or optimizer updates. A
copy-on-write candidate is admitted only after an independent held-out
one-step prediction, recursive rollout, and source-retention probe.

Run one seed with:

```bash
.venv/bin/python -m experiments.policy_free_factual_residual_growth.train \
  --seed 101 \
  --report-out /tmp/policy-free-factual-residual-101.json
```

The matched full-model-copy and fresh-model controls are intentionally kept:
they can fit the successor, but their source-retention curve is expected to
fail because they overwrite or replace the shared computation. This promotes
one-pass factual residual acquisition and retention, not general continual
learning, unrestricted residual capacity, arbitrary program induction, or
policy learning.

# Nonlinear goal alignment growth — promoted

This four-seed audit tests copy-on-write growth of a nonlinear goal-memory
alignment without replay. An initial 16-feature adapter consumes `24` sparse
alignment pairs and fails held-out verification. It then grows to `80` frozen
random features after a retention check, consumes `24` new alignment pairs,
and is evaluated on `48` held-out pairs.

Initial mastery was `0.625`, `0.392`, `0.483`, and `0.450`. After growth,
mastery was `1.000`, `0.975`, `0.992`, and `0.950`. Post-growth held-out MSE
was between `0.00062` and `0.00284`, while retention error at the growth seam
was below `2e-11` on every seed.

The old adapter behavior was retained copy-on-write, the adapter and old goal
verifier memory were unchanged during search, persistence was exact, and both
old alignment rows and verifier outcomes had zero replay. This promotes a
bounded external nonlinear capacity-growth transaction, not unrestricted
memory growth or general continual learning. The next pressure test is
capacity pressure with multiple concurrent frontends and quarantine/eviction
when growth is refused.

Reproduce one seed with:

```bash
PYTHONPATH=src:. .venv/bin/python \
  experiments/external_goal_representation_nonlinear_growth/train.py \
  --seed 84601 \
  --report-out /tmp/external-goal-representation-nonlinear-growth-84601.json
```

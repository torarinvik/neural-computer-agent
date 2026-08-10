# Learned external admission cost — 2026-08-10

This audit replaces hand-specified transfer/fresh deployment costs with an
independently versioned memory-side learner. The learner sees only masked
opaque context values, the selected verified source's coverage, and the
current external-cell count. After a branch completes, it updates only the
selected transfer or fresh head from normalized continuation work. It stores
no task labels, trajectories, or replay examples, and the controller and
state adapter remain frozen.

All three seeds pass the existing source-selection, verifier, retention,
complete-prefix, reversal, corruption, persistence, causal-control, and
zero-replay gates. Every admission has a cost estimate and every model state
round-trips exactly through its own checksum boundary. The learned policy
selects `transfer -> fresh -> transfer` in every warm run. The matched-fresh
run for seed `85302` selects `fresh -> fresh -> transfer`, demonstrating that
the new layer does not blindly preserve the historical sequence when the
outcome-only challenger favors a fresh candidate.

| seed | warm selection | matched-fresh selection | warm/fresh unique bits |
| ---: | --- | --- | ---: |
| 85301 | `transfer / fresh / transfer` | `transfer / fresh / transfer` | `91 / 89` |
| 85302 | `transfer / fresh / transfer` | `fresh / fresh / transfer` | `84 / 93` |
| 85303 | `transfer / fresh / transfer` | `transfer / fresh / transfer` | `85 / 82` |

The cost model receives three replay-free observations per run (two transfer,
one fresh in most runs; the seed-85302 fresh control receives one transfer and
two fresh observations). Reports include the prediction traces, selected
branch errors, model summaries, and model-state digests.

Reproduce from the repository root:

```bash
PYTHONPATH=. .venv/bin/python \
  -m experiments.policy_free_intention_routing.sequential_admission \
  --seed 85301 \
  --learned-cost \
  --report-out /tmp/policy-free-intention-learned-cost.json
```

This promotes a bounded learned admission-cost contract and removes a
caller-side cost schedule. It does not establish broad task-family
generalization, universal positive transfer, arbitrary new computation,
unrestricted memory growth, compression, or general continual learning.

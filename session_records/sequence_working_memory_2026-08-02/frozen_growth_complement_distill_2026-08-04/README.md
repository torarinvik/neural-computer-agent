# Frozen-controller external-growth acquisition

This is the first replicated end-to-end acquisition result for the isolated
growth-state boundary. A parent that already solves the mixed forward/reverse
working-memory frontier receives a zero-output generic successor slot. The
shared controller stays frozen; only the successor state is trained from
rendered events, opaque attempted actions, and scalar verifier outcomes.

The learned slot is written as tensor-only state to
`neural_computer.ExecutableArtifactMemory`, reloaded in a fresh model
instance, and checked against the live child. Parent-logit distillation on
fresh rehearsal streams protects inherited behavior. It is a trainer-only
stability mechanism; the artifact store does not interpret or execute the
payload.

Command used for both the promoted candidate and its replication:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python -m \
  experiments.working_memory_continuous.acquire_frozen_growth \
  --parent artifacts/checkpoints/span8_addressed_parent_scale1_seed32001.pt \
  --memory session_records/sequence_working_memory_2026-08-02/frozen_growth_complement_distill_2026-08-04/memory \
  --report session_records/sequence_working_memory_2026-08-02/frozen_growth_complement_distill_2026-08-04/report.json \
  --seed 61004 --steps 256 --batch-size 16 --audit-count 64 \
  --target-span 2 --target-operation complement --rehearse-spans 4 6 8 \
  --distractors 2 --distill-old-weight 1.0 --device cpu
```

| Seed | Parent | Child | Zeroed growth | Target gain | Retention changes (4/6/8) |
| ---: | ---: | ---: | ---: | ---: | --- |
| 61004 | 42.19% | 74.22% | 42.19% | +32.03 pp | 0.00 / −0.26 / −0.20 pp |
| 61005 | 41.41% | 75.00% | 41.41% | +33.59 pp | 0.00 / −0.78 / 0.00 pp |

Both runs used 4,096 unique logical lifetimes, 20,480 unique verifier bits,
256 optimizer updates, and no replayed examples. The child and fresh
rehydrated model matched exactly on the measured audits; the frozen core was
bit-identical before and after growth loading; and SHA-256 corruption was
rejected. Both runs pass the initial-insertion, causal-growth,
rehydration, retention, and corruption gates.

The unprotected 256-update control reached the target but lost 3.91 points on
span eight. A same-budget success-only objective produced no target gain.
Those controls identify parent-behavior distillation as the useful retention
repair at this rung rather than merely increasing exposure or changing the
bandit loss. The reward-shuffled control gained `+7.03` points and stayed
artifact-causal, but was far below the aligned-outcome gain. Therefore this
record does not claim that the target gain is exclusively attributable to
correct reward correspondence; reward attribution and address discovery remain
open controls.

This promotes only a narrow result: an adjacent synthetic working-memory
procedure can be acquired into isolated external growth state while a shared
controller remains frozen, with bounded retention across two seeds. It does
not establish cold-start address discovery, arbitrary procedure discovery,
variable-capacity mastery, exclusive reward attribution, natural language, or
general cognition.

Reports:

- `report.json` — seed 61004;
- `../frozen_growth_complement_distill_replication_2026-08-04/report.json` —
  seed 61005;
- `../frozen_growth_complement_rung3_2026-08-04/report.json` — unprotected
  retention-failure control;
- `../frozen_growth_complement_success_only_2026-08-04/report.json` —
  same-budget objective control;
- `../frozen_growth_complement_distill_reward_shuffled_2026-08-04/report.json`
  — reward-shuffled attribution control.

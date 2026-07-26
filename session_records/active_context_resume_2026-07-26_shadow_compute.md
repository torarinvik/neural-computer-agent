# Resume: sample-efficient shadow compute allocation

## Current verified state

Branch: `work`

The latest completed experiment is a passive compute allocator that learns
whether one extra external-memory read is worthwhile. It is trained only from
four controller-created read statistics, a uniformly logged attempted
read/no-read action, exact propensity `0.5`, a normalized read cost `0.01`,
and the attempted action's scalar verifier outcome.

No correct compute action or unattempted outcome enters training.

The direct inverse-propensity advantage objective passed and replicated with a
105-parameter head at 720 unique verifier bits:

- seed 7424: 69.0% choice accuracy, 59.7% oracle-gap capture;
- seed 7425: 70.6% choice accuracy, 60.2% oracle-gap capture.

Both reached the primary stable threshold at the first measured 120-bit
prefix. Reward/feature/zero-evidence controls failed; evidence shuffling made
utility worse than always-read. Old skills and persistence passed.

The inherited 49-parameter gate is still stronger and must not be overwritten.
Its matched audit is saved at:

`session_records/shadow_compute_allocation_2026-07-26/inherited_gate_audit_7424_7425.json`

## Exact next experiment

Test whether the near-matched 57-parameter width-8 head only needed a little
more experience:

```bash
.venv/bin/python -m \
  experiments.unified_cognitive_controller.train_shadow_compute_advantage \
  --parent-checkpoint \
  artifacts/checkpoints/unified_memory_online_utility_seed6810.pt \
  --selected-prefix \
  artifacts/checkpoints/balanced_maximin_stream7085_clone7211_round54.pt \
  --report \
  session_records/shadow_compute_allocation_2026-07-26/advantage_width8_24_seed7426.json \
  --device cpu --seed 7426 --head-hidden 8 --steps 24
```

This changes only experience: 720 → 1,440 fresh bits. Do not alter gates,
learning rate, capacity, read cost, or controls.

- If it passes, replicate unchanged with seed 7427.
- If it fails, stop this capacity fork and retain the replicated
  105-parameter blueprint.
- Do not replace the inherited gate weights on this mastered task.

## Important files

- `experiments/unified_cognitive_controller/train_shadow_compute_advantage.py`
- `experiments/unified_cognitive_controller/train_shadow_compute_critic.py`
- `experiments/unified_cognitive_controller/audit_inherited_compute_gate.py`
- `session_records/shadow_compute_allocation_2026-07-26/README.md`
- `experiments/forward_transfer_attention/SAMPLE_EFFICIENCY_LEDGER.md`

No training process was running when this session was saved.

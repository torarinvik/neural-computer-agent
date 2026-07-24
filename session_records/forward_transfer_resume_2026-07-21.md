# Forward-transfer neural computer — continuation note

Saved 2026-07-21 from `elisa-screenwatch`.

## Current milestone

The frozen ~2.14M-parameter sensory neural computer and a 724,001-parameter latent-only
consolidator receive RGB/PCM and learned memory only—never mappings, rules, task IDs, or game
state. After every support observation, recursive consolidation reduces the current memory plus the
new write back to one row.

Four-seed held-out consolidation result (1,024 lifetimes per seed):

- compact one-row AUC: 45.93%;
- full five-row AUC: 43.57%;
- one-shot gain: +4.71 points, positive on every seed;
- two-shot gain: +5.37 points, positive on every seed;
- final old-task accuracy: 47.73%, versus 42.87% initially;
- corrupt compact rows reduce post-support performance to chance.

One seed-23 consolidator transfers without retraining across all four independently trained
controllers. On 1,024 lifetimes per controller it scores 46.41% AUC versus 44.18% full-memory AUC,
with a +4.62-point one-shot gain on every controller.

The same frozen universal consolidator and controllers transfer without training from spatial
attention (left/right) to feature attention (circle/square):

- zero-shot: 42.86%; one-shot: 47.67% (+4.80 points);
- compact AUC: 46.22%; full-memory AUC: 43.94%;
- final old-task accuracy: 47.20%, versus 43.95% initially;
- 256-lifetime causal audit at one shot: intact 47.58%, empty 12.01%, shuffled 13.33%,
  garbage 11.77% (chance 12.5%).

All four controllers improved. Cue/rule assignments use independent BLAKE2b hashes, including a
4,096-lifetime leak test for each primitive.

## Important artifacts

- `experiments/forward_transfer_attention/README.md`: design and exact conclusions.
- `consolidator.py`, `train_consolidator.py`: recursive latent consolidation.
- `targeted_decorrelated/seed_*.pt`: four frozen controller checkpoints.
- `targeted_consolidator_replication/seed_*.pt`: four consolidator checkpoints.
- `targeted_consolidator_replication/summary.json`: primary replicated summary.
- `targeted_consolidator_causal_audit/`: memory intervention reports.
- `targeted_cross_controller_matrix/`: 4x4 adapter/controller screening matrix.
- `targeted_universal_candidate/`: full-scale universal seed-23 adapter evaluation.
- `targeted_cross_primitive_confirmation/`: full-scale unseen shape-attention results.
- `targeted_cross_primitive_causal/`: unseen-task intervention audit.

## Cloud instance

The last Vast instance was accessed with:

```sh
ssh -i /Users/torarinvikbjarko/.ssh/id_ed25519_vast_ai \
  -p 17111 root@89.221.67.164 -L 8080:localhost:8080
```

All important outputs have already been downloaded. The instance is not required to resume.

## Next high-ROI experiment

Add a third deterministic primitive requiring temporal selection, such as attend to the object that
appeared first or last. Follow the established staged gate:

1. validate determinism, unique answers, sensory-only interfaces, and cue independence;
2. run 256 frozen lifetimes with the universal seed-23 consolidator on four controllers;
3. scale to 1,024 only if one-shot gain is positive across controllers;
4. run empty/shuffled/garbage interventions;
5. only fine-tune if frozen transfer fails, and then audit spatial and shape regression.

## Verification at save time

`python -m pytest experiments -q` passed 142 tests and 15 subtests. `git diff --check` passed.
The worktree was intentionally not committed because it contains unrelated pre-existing user
changes and other untracked experiments.

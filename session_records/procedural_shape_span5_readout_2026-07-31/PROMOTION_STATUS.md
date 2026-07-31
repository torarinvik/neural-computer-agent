# Promotion status

**Baseline span-five capability: replicated pass.** Two independent replay
seeds pass the baseline gates at nuisance level 0.135.  The selected candidate
is `artifacts/checkpoints/unified_procedural_shape_span5_replay_seed44906.pt`.

**Robust graduation: not yet passed.** At nuisance level 0.17, the candidate
scores 90.53% overall, 84.74% strict conflict, and 92.71% old span-three
retention.  The next run should mix the current level with 0.17 gradually,
then repeat the same two-seed audit.  No architecture change is justified:
the five-item relation is already present and behaviorally learnable.

The replay runner now rejects adjacent nuisance increments larger than 0.0001
by default. The first stable three-step staircase (0.1350 → 0.1352) passed its
endpoint gates using learning rate `3e-4` and four rehearsal updates. The new
candidate is `artifacts/checkpoints/unified_procedural_shape_span5_micro1352_seed45301.pt`.
Reaching 0.17 still requires a 350-step staircase from 0.135; an interrupted
coarse run is not treated as evidence.

The following microscopic rung, 0.1353 → 0.1355, also passed: 95.80% overall,
93.25% strict conflict, 50.88% reset accuracy, and 98.96% old retention. Its
candidate is `artifacts/checkpoints/unified_procedural_shape_span5_micro1355_seed45401.pt`.

The 0.1356 → 0.1358 rung also passed: 96.58% overall, 96.01% strict
conflict, 49.41% reset accuracy, and 99.35% old retention. The new candidate is
`artifacts/checkpoints/unified_procedural_shape_span5_micro1358_seed45501.pt`.

An adaptive gate then tested 0.1359, 0.1360, and 0.1361. Every level passed
before training, so the run spent zero optimizer updates. The full 0.1361 audit
passed with 95.90% overall, 95.53% strict conflict, 48.83% reset accuracy, and
99.35% old retention. This is the current sample-efficiency frontier; the
adaptive candidate is
`artifacts/checkpoints/unified_procedural_shape_span5_adaptive1361_seed45601.pt`.

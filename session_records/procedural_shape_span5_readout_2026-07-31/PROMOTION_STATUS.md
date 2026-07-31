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
by default. Reaching 0.17 therefore uses a 350-step staircase from 0.135,
rather than a single jump; an interrupted coarse run is not treated as
evidence.

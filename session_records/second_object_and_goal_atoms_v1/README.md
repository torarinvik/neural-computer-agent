# F226: selection quality was the goal-search bottleneck (2026-08-13)

Probes: experiments/games_amodal/probes/second_object.py (pair goal
language, v1 density worlds -> v2 +intercept/lowest-reduction -> v4
robust selection) and experiments/games_amodal/probes/goal_atoms.py
(coordinate-atom goal language, v3 -> v4). Registered predictions and
per-version outcomes are in the probe docstrings; the narrative and
statistics are in docs/MEMORY_BANK_DESIGN.md under F226.

Result dirs (gzipped raw JSON, per seed):
  so/   v1 pairs, 3 seeds, density worlds only
  so2/  v2 pairs, 6 seeds, +intercept worlds, +low reductions,
        selection 32x10 single stream
  ga/   v3 atoms, 6 seeds, selection 32x10 single stream
  ga4/  v4 atoms, 6 seeds, selection 48x12 min-of-two-streams
  so4/  v4 pairs, 6 seeds, selection 48x12 min-of-two-streams

Headline paired statistics (best arm, evaluation stream seed*977):
  pairs v4 - pairs v2, pooled 8 worlds x 6 seeds: +0.093, t=+2.61
  intercept2 pairs v4: mean +0.195 (was -0.865 at v2 budget);
    vs random t=+20.8
  atoms v4 - pairs v4, pooled: -0.064, t=-2.65 (pair grammar is a
    useful prior at matched selection quality)
  eight-six arm difference, pooled, both languages, all versions:
    |t| < 1 everywhere -- the width null is final for this family.

Analysis scripts used during the session: so_show.py (scratchpad),
plus inline python in the transcript; all numbers above recompute from
the JSONs with pairing per (seed, world).

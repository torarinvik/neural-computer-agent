# Perception by control (F213-F215)

F213: programmability x coverage criterion; broken twice by its own
search (avatar-only degeneracy, then duplicate slots). discover*.json.gz.

F214: goal chosen per world from reward rediscovers the human goal on
7/8 worlds (pc3). Perception discovery still failed under prediction
criteria. Two of my bugs caught by diagnostics (ABSENT masking,
cross-world presence averaging).

F215: criterion = the environment's return. pbc2/pbc-*.json.gz holds
the comparable evaluation (all 14 candidates, all 8 held-out worlds,
no-goal worlds scored at random's return), refine-*.json.gz the
full-fidelity top-5 re-ranking.

    Spearman(train control, held-out margin): +0.754 +0.921 +0.899
    pipeline - handwritten per seed:  +0.0014 +- 0.0584  t=+0.02
    paired per (seed,world) n=24:     +0.0013 +- 0.1496  t=+0.01
    no-selection null:                +0.3668 (selection adds +0.5571)

pbc/ (first run) kept because its world-skipping inflation is the
recorded artifact: vocab_peaks +1.3667 on 5 easy worlds, vocab_random3
+0.72 -> +0.07 under honest accounting.

Reproduce: python -m experiments.games_amodal.probes.perception_by_control
--seed S ; then pbc_refine.py S <scratch>. Single-threaded.

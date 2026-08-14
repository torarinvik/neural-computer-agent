# reward_vs_dynamics_v1 (F251)

Probe: experiments/games_amodal/probes/reward_vs_dynamics.py
Seeds: 69316 1234 4242 555 31337 2718 (6).

Privileged one-cell split of F250's residual: true-dynamics depth-2
planning with true vs learned edge rewards (learned V both arms).
Learned reward collapses to random on all worlds incl. control
(+1.50 -> -0.07): reward binding indicted, dynamics exonerated;
consumption-jump artifact identified (proximity V punishes eating
over true dynamics). See F251 in docs/MEMORY_BANK_DESIGN.md.

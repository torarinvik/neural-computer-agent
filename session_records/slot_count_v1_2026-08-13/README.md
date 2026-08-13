# Slot count and the co-scaling law (F225)

sc-*: SLOTS=8, single-term signed goals. Plant gates 1.0 x3 at 40k.
8-slot minus archived 6-slot reference (same protocol/seeds): -0.1777
t=-3.00; goals name new slots 1/24.

sc2-*: same with F222 composite search. Recovers to 6-slot parity
(+0.0319, t=+0.37); composite - single at 8 slots +0.2096, t=+3.39.
New slots still used 0/24; multi-object -0.19 below 6-slot: this world
family never needed the width (avoid2 at 92-95% ceiling per F220).

Law: interface width must co-scale with goal-language width, or width
is a cost. Reproduce: python -m experiments.games_amodal.probes.slot_count
--seed S [--composite]

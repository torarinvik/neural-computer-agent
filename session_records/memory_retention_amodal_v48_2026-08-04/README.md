# Extended fresh-parent transfer-control rejection

v48 repeats the seed-17 transfer diagnostic with a matched 2,048-step
phase-1 budget and 2,048 retention steps for both the transferred and fresh
learners. The transferred learner again reaches stable threshold at `28,672`
bits. The fresh learner still fails parent qualification after all 2,048
phase-1 updates, so its transfer denominator remains undefined.

The short transfer budget is therefore not the only bottleneck. Do not claim a
population transfer ratio until fresh-parent qualification is treated as an
explicit gate and tested across multiple fresh initialization seeds.

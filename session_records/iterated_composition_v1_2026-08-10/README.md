# Iterated composition (F121)

Probe 221. Same setting as F119 (1 world, ignorance off, 30k updates,
dim 128); only the program interface changes — a latent stepped once
per token through a SHARED step function instead of one-shot.

  one-shot (mc-fit-69316.json): trained 1.0000, held-out 0.0794
  iterated (it-fit-69316.json): trained 1.0000, held-out 1.0000

Chance 0.0435. No intermediate supervision; identical parameter count.
The shared step function is the entire compositional prior.

Replication (added 2026-08-10): seed 69317 also 1.0000 / 1.0000.

Cross-ground replication (F133): boolean single-world control gives
1.0000 exact and 1.0000 per-bit, identical to the arithmetic result.

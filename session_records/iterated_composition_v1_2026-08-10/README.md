# Iterated composition (F121)

Probe 221. Same setting as F119 (1 world, ignorance off, 30k updates,
dim 128); only the program interface changes — a latent stepped once
per token through a SHARED step function instead of one-shot.

  one-shot (mc-fit-69316.json): trained 1.0000, held-out 0.0794
  iterated (it-fit-69316.json): trained 1.0000, held-out 1.0000

Chance 0.0435. No intermediate supervision; identical parameter count.
The shared step function is the entire compositional prior.

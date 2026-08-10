# Oracle substitution: the outcome model is the binding constraint (F110)

Replace the outcome model with ground truth INSIDE the search; keep the learned
dynamics and beam intact. One run, cannot be argued with — the only thing
changed is the quantity under suspicion.

| arm | held-out reward |
| --- | ---: |
| learned outcome model (F109 best) | -0.0205 |
| ORACLE outcome, learned everything else | **+0.1234** |
| hand-coded oracle policy (ceiling) | +0.1954 |
| best context-free policy (floor) | -0.0318 |

Per-seed +0.1093 / +0.1375.

## The measured decomposition of the games gap

- entry not read (fixed, F107): +0.0499
- object hallucination in search (fixed, F109): +0.0078
- **outcome model inaccuracy: +0.1439 — dominant**
- search + dynamics residual: +0.0720, unaddressed

## Target for future work

+0.1234 is what perfect values buy through this search. Any outcome-model
improvement (more visits per world, better labels, an n-step value head) should
be scored against that number, on this benchmark, with the twin controls.

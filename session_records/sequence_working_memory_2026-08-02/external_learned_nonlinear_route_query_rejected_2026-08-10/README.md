# Rejected learned nonlinear route-query proposal (2026-08-10)

This audit applies the exported session's richer `trajectory_stats` idea to
the rejected replay-free learned-MLP factual-memory experiment. A separate
versioned `ExternalTransitionRouteQuery` proposes an opaque slot using
slot-local copy-on-write address adapters and a richer route key consisting of
the projected context, final recurrent state, mean recurrent state, and max
recurrent state. The proposal has a minimum cosine-quality floor, but factual
prediction error remains the independent acceptance gate.

The route query was enabled with:

```text
.venv/bin/python -m experiments.external_learned_nonlinear_open_world.train \
  --seed <seed> --route-query --match-tolerance 0.01 \
  --route-query-minimum-score 0.80 \
  --report-out report_seed<seed>.json
```

Four nonlinear regimes exposed `48/64` rows. The controller and base context
encoder remained frozen; candidate rows were consumed through
`streaming_gradient`, with no old-regime replay and no raw candidate rows
retained. The route query was proposal-only: it could not override factual
verification.

## Result

| seed | all held-out quality | revisit matches | promoted |
| ---: | :---: | ---: | :---: |
| 82601 | pass | 0/6 | no |
| 82602 | fail | 1/6 | no |
| 82603 | pass | 0/6 | no |

The route proposal sometimes selected the correct slot, but the learned MLP
still failed the strict factual verification gate. A looser factual tolerance
was also tested during smoke: it admitted a novel regime as an existing slot,
so it is not an acceptable workaround. The initial compact cosine query was
also diagnostic-only: its scores collapsed near `0.99–1.00` across slots.

## Interpretation

The export's trajectory-statistics route query is promising infrastructure,
but it is not sufficient when copied into this boundary as a frozen similarity
metric. Learned identity calibration is still missing. The next candidate
must learn a reusable route score from verifier-grounded counterfactuals or a
meta-learned representation, while retaining a factual challenger and novel-
regime rejection path. More threshold tuning is rejected.

This rejects the current cosine/slot-local proposal for learned nonlinear
open-world routing. It does not reject trajectory statistics, external
route-query interfaces, or model-first continual learning generally.

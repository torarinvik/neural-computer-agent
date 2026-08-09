# Rejected learned route-query scorer for nonlinear external memory (2026-08-10)

This audit applies the exported session's model-first, external-memory idea to
identity routing. A frozen controller and frozen base context encoder feed a
separate `OpaqueCandidateGrowthRouter`. The scorer sees only opaque trajectory
statistics and slot keys, and is updated from current-window factual
counterfactual errors. It receives no old-regime replay and retains no raw
transition rows. Factual transition models remain the authority for matching.

The experiment was run with:

```text
.venv/bin/python -m experiments.external_learned_nonlinear_open_world.train \
  --seed <seed> --learned-route-query --learned-route-updates 128 \
  --match-tolerance 0.01 --report-out report_seed<seed>.json
```

## Result

| seed | held-out quality | route proposal matches factual winner | revisit matches | promoted |
| ---: | :---: | :---: | :---: | :---: |
| 82601 | fail | no | 0/6 | no |
| 82602 | fail | no | 0/6 | no |
| 82603 | fail | no | 0/6 | no |

The scorer rapidly became biased toward the newest slot. Its proposal was
often wrong for older regimes, despite exact persistence and a frozen
controller. A factual fallback was added and regression-tested so the scorer
cannot block a verifier-established match; this preserves safety but does not
turn the scorer into a learned capability gain.

## Interpretation

The failure is the target continual-learning lesson: a shared trainable route
scorer updated only from the newest evidence window forgets old route
identities. The exported session's route-learning strategy transfers only when
the route learner itself has a protected memory of old constraints. The next
candidate should therefore use isolated per-slot route state or a compressed,
verifier-maintained route-constraint memory, with a challenger for novel
regimes. More updates on the same current window and threshold tuning are not
valid fixes.

This rejects the shared current-window learned route scorer. It does not reject
external transition memory, trajectory statistics, model-based search, or the
proposal-only route-query interface.

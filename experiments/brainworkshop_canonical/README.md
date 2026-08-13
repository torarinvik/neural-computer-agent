# Canonical Brain Workshop frontier

This directory is the retained n-back and working-memory research surface.

```text
rendered symbol -> learned amodal event -> fixed controller
                -> external indexed history and compute file
                -> intention -> opaque keypress decoder
                <- trusted scalar outcome event
```

The verifier privately owns the n-back target. The learner receives learned
events, its opaque actions and propensities, external memory, and scalar
outcomes. It never receives the horizon as a semantic task ID or a correct
action label.

## Promoted frontier

`external_compute_append_only_depth_growth.py` is the current bounded working-
memory result. A frozen source file masters n-back-16, then a fresh external
file learns n-back-32 while the source, controller, and event frontend remain
unchanged.

Across seeds 17 and 18:

| Measurement | Seed 17 | Seed 18 |
| --- | ---: | ---: |
| n-back-16 before extension | 1.0000 | 0.9992 |
| n-back-32 new file | 1.0000 | 1.0000 |
| n-back-16 retention | 1.0000 | 0.9992 |

Missing/corrupted history, shuffled actions, and fresh shuffled-outcome
training all failed the 0.80 mastery gate. Each seed used 512 learned-file
optimizer updates, 163,840 verifier bits, and zero replay.

Run it with:

```bash
PYTHONPATH=src .venv/bin/python \
  -m experiments.brainworkshop_canonical.external_compute_append_only_depth_growth
```

Evidence:
`session_records/brainworkshop_append_only_nback32_depth_promoted_2026-08-13/`.

## Current bottleneck

The external compute/artifact lifecycle is reliable, but inherited eviction
knowledge has not transferred to a held-out n-back family. The current active
audit tests a neutral probationary fallback. Until it passes replicated fresh
controls, the correct policy is to retain the architecture and reset inherited
maintenance weights.

The next n-back campaign must measure stable bits-to-threshold against a matched
fresh learner. Raising final accuracy alone is not the objective.

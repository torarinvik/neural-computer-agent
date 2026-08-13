# External depth-file route reversal (2026-08-13)

Status: **promoted bounded replay-free external computation growth**.

This audit composes the indexed-history n-back-16 and n-back-32 boundaries.
The controller and learned event frontend first support an opaque n-back-16
file. A second opaque file learns n-back-32 with a 32-slot relative-age
history. Both files are then frozen while route memory learns which file to
select from the learned event tensor and scalar terminal episode outcomes.

The route is deliberately reversed behind the source cue: the new n-back-32
file becomes correct for that cue, while the old n-back-16 file remains
directly executable. A fresh unknown cue must conservatively fall back to the
oldest file and must not acquire the new task. A batch-shuffled outcome
control tests that route mastery is causal rather than an artifact of the
probe schedule.

Both independent seeds passed every gate:

| Measure | Seed 17 | Seed 18 |
| --- | ---: | ---: |
| Source n-back-16 direct accuracy | 1.0000 | 1.0000 |
| Target n-back-32 direct accuracy | 0.9344 | 0.9344 |
| Target routed accuracy | 0.9375 | 0.9375 |
| Same-cue replacement accuracy | 0.9297 | 0.9297 |
| Old-file forced retention | 1.0000 | 1.0000 |
| Unknown-cue accuracy | 0.6125 | 0.6125 |
| Shuffled-route target accuracy | 0.6312 | 0.6312 |
| Replayed examples | 0 | 0 |

Each seed used 256 source updates, 256 target updates, 256 route updates,
eight route-calibration lifetimes, eight transition probes, and two held-out
retention lifetimes. The controller and event frontend stayed byte-identical;
both external files stayed byte-identical during route learning; serialization
and reload reproduced the changed route exactly.

This promotes automatic selection and failure-driven same-cue replacement
between two bounded external computation files. It does **not** establish
unrestricted memory growth, learned allocation or compression, arbitrary
program induction, or general continual learning. The remaining bottleneck is
open-ended allocation and lifecycle management for a growing file bank without
hand-selected slots or a fixed two-file experiment.

Implementation: `experiments/brainworkshop_canonical/external_compute_depth_route_reversal.py`.
The full promotion command is:

```text
PYTHONPATH=src uv run python -m experiments.brainworkshop_canonical.external_compute_depth_route_reversal --report-out <path> --seed <17-or-18>
```

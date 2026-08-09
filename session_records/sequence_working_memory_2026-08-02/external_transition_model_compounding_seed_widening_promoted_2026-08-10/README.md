# Policy-free model compounding — seed widening promoted

This archive widens the promoted factual-transition-model compounding rung
from two to four seeds. The mechanism is the policy-free route imported from
the exported game-learning session: external models store transition facts,
and `ExternalModelBasedPlanner` derives intentions from the current opaque
goal. No task policy is stored in the controller.

| seed | warm target updates | fresh target updates | warm cumulative | fresh cumulative |
| ---: | ---: | ---: | ---: | ---: |
| 70311 | 24/23/17 | 38/38/34 | 1,264 | 1,310 |
| 70312 | 25/22/17 | 35/38/41 | 1,264 | 1,314 |
| 70313 | 25/28/21 | 42/40/31 | 1,274 | 1,313 |
| 70314 | 34/30/23 | 46/48/43 | 1,287 | 1,337 |

The two new seeds passed every existing gate: all warm and fresh targets
reached deployed planner mastery, each warm target used fewer updates than its
matched fresh control, prior models stayed at mastery and byte-stable, the
controller stayed frozen, planner search was inference-only, and old-regime
replay during target adaptation was zero. Zero-shot mastery remains reported
separately (`0.0/0.667/1.0` and `0.333/0.667/0.667` for seeds 70313/70314).

Claim boundary: this is a four-seed replicated downward acquisition-cost signal
for one small nested dynamics family. Context vectors are supplied, the
planner horizon is finite, and the result is not general continual learning,
unrestricted memory growth, or arbitrary task transfer. The next pressure
test must keep the same accounting while stressing partial evidence, gradual
drift, or genuinely broader dynamics.

The original two reports remain in
`../external_transition_model_compounding_promoted_2026-08-09/`; this archive
contains the two new reports and their checksums.

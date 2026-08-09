# Accounted policy-free model compounding

This pressure test turns the exported games-session lesson into a canonical
audit. A policy stores preferences that can become wrong on a new task; an
external transition model stores factual dynamics, and `ExternalModelBasedPlanner`
derives behavior from the current opaque goal at inference time.

The audit therefore reports four quantities separately:

- zero-shot capability before target adaptation;
- target model updates needed to reach the mastery threshold;
- cumulative model updates charged from the first source regime;
- inference search expansions and latency.

Each new regime is initialized from the immediately preceding external model
slot and adapted in isolation. Earlier slots are re-evaluated and their bytes
are checked after every later phase. No controller parameters or old-regime
observations are updated or replayed during target adaptation. Fresh target
models provide the matched acquisition controls.

This is deliberately a fast validation rung, not a general-continual-learning
claim. It uses one small nested dynamics family and a finite transition bundle.
The result is only promoted if every warm regime reaches mastery, earlier
models retain mastery, warm adaptation beats fresh adaptation at every target,
and the cumulative cost is reported without hiding source acquisition.

```text
.venv/bin/python experiments/external_transition_model_compounding/train.py \
  --seed 70311 \
  --report-out /tmp/external-transition-model-compounding.json
```

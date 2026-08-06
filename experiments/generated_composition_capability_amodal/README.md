# Generated compositional capability pilot

This pilot trains one fresh external capability against a frozen parent
controller on `generated_composition`. Each lifetime samples a two-primitive
program from a verifier-private grammar and renders the two primitive cues as
ordinary learned events. The controller receives no program ID, primitive
labels, correct actions, or verifier-private composition metadata.

Run the short rung with:

```bash
PYTHONPATH=src uv run python -m experiments.generated_composition_capability_amodal.train \
  --parent-updates 32 --updates 64 --batch-size 8 --audit-count 16 \
  --eval-every 16 --report-out /tmp/generated-composition/report.json
```

This is an acquisition diagnostic, not a continual-learning promotion. The
full promotion must append the generated capability behind the protected
route chain and retain every previously mastered artifact.

The serial external-stack diagnostic is:

```bash
PYTHONPATH=src uv run python -m experiments.generated_composition_capability_amodal.train_pipeline \
  --parent-updates 32 --updates 64 --program-count 2 --batch-size 8 \
  --audit-count 16 --eval-every 16 \
  --report-out /tmp/generated-composition-pipeline/report.json
```

The routed stack uses the new external composition binder:

```bash
PYTHONPATH=src uv run python -m experiments.generated_composition_capability_amodal.train_pipeline \
  --stack routed --parent-updates 32 --updates 64 --program-count 2 \
  --batch-size 8 --audit-count 16 --eval-every 16 \
  --report-out /tmp/generated-composition-routed/report.json
```

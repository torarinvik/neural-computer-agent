# External outcome-program router

This pressure test connects terminal scalar credit to the canonical external
program boundary. A shared register interpreter executes opaque
`ExternalProgramArtifact` values from `ExternalSequenceProgramMemory`; an
`ExternalOutcomeProgramRouter` selects a sequence of program slots and learns
from one terminal verifier outcome. The controller and execution components
are frozen during route acquisition.

Run a matched seed with:

```text
.venv/bin/python experiments/external_outcome_program_router/train.py \
  --report-out /tmp/external_outcome_program.json \
  --seed 69316
```

The experiment deliberately reports the pretraining updates used to admit the
shared executable interpreter separately from the zero-optimizer-update
outcome-only route learning. It is a bounded execution/credit result, not a
claim of program induction or general continual learning.

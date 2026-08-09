# External model-based planner

This pressure test implements the direction identified in the exported
continual-learning session: store learned transition facts externally and
derive behavior by inference-time search instead of storing a task-specific
policy in the frozen controller.

Run it with:

```text
.venv/bin/python experiments/external_model_based_planner/train.py \
  --seed 69316 \
  --report-out /tmp/external-model-based-planner.json
```

The fixture uses opaque learned state and intention tensors.  A frozen
`AmodalCognitiveController` is included only for the isolation audit; all
updates go to the replaceable `ExternalTransitionModel`.  The target sequence
receives zero optimizer updates and zero replayed examples.  The report also
contains fresh-model, goal-shuffled, transition-shuffled, persistence, and
prefix-retention controls.

This is an initial bounded rung.  It demonstrates external factual transition
learning and computed behavior on a tiny deterministic surface; it is not yet
general continual learning, unrestricted model growth, or arbitrary program
induction.  Promotion requires matched multi-seed replication and a genuinely
different dynamics family.

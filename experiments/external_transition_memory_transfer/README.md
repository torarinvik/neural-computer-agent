# Disjoint-dynamics transition-memory transfer

This pressure test extends the external model/search result with an
append-only factual transition store. Two dynamics regimes use the same opaque
state and intention representations but different opaque context tensors, so
their facts must coexist without overwriting each other. A learned scalar
verifier scores terminal states during inference-time search.

Run it with:

```text
.venv/bin/python experiments/external_transition_memory_transfer/train.py \
  --seed 69401 \
  --report-out /tmp/external-transition-memory-transfer.json
```

The target phase appends new transition facts, performs zero controller or
transition-model optimizer updates, and replays zero source examples. The
report includes source-retention, shuffled-goal, shuffled-context, corrupted
memory, fresh-memory, and persistence controls.

This is still a bounded nonparametric continual-memory rung. It does not show
unrestricted memory growth, learned context discovery, autonomous consolidation,
or general continual learning.

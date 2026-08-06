# Sequential external-capability append

This audit pressure-tests the memory boundary after the promoted
three-program external-capability bank. It fills a capacity-three bank with
protected rows, attempts a fresh `rotate4` append, requires the protected write
to fail explicitly, grows the bank transactionally, and then trains only a
fresh memory-side route extension.

The parent controller and the old three-row router remain frozen after the
append point. The capability program and its decoder are replaceable external
state. Route keys are learned opaque event-derived addresses; random-key
control is retained as a rejected diagnostic because the router cannot infer an
arbitrary query-to-random-ID mapping at the tested budget.

Run from the repository root as a module:

```bash
uv run python -m experiments.sequential_external_capability_amodal.train \
  --report-out /tmp/sequential-external-capability/report.json
```

Promotion requires two seeds to pass stable-prefix capability mastery,
protected-capacity rejection, transactional growth, old/new route accuracy,
candidate permutation, shuffled-outcome, causal wrong-artifact, reload,
corruption, frozen-digest, and zero-replay gates. The resulting claim is one
protected sequential append, not general continual learning.

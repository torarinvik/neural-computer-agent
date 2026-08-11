# Causal external working-memory transfer — 2026-08-11

This audit closes a measurement flaw in the earlier fast-cell transfer probe.
The versioned `ExternalWorkingMemoryCell` reads the current learned event
against the old external state before the keypress is selected, then appends
the current event/action/outcome row. A source codec is trained from fresh
scalar verifier outcomes on n-back-2 and frozen. Evaluation uses fresh
external state and a matched untrained cell.

| seed | source mastery | frozen fresh-state n-back-2 | fresh control | shuffled outcomes | history reset | n-back-3 probe |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | 1.0000 | 1.0000 | 0.5000 | 0.4824 | 0.5000 | 0.4990 |
| 18 | 1.0000 | 1.0000 | 0.5000 | 0.4824 | 0.5166 | 0.4922 |

Both seeds pass source mastery, causal fresh-state mastery, fresh-control,
shuffled-outcome, history-reset, frozen-controller, frozen-codec, and
zero-replay gates. This is a causal promotion: the learned memory computation
is used before action selection, not merely reconstructed after the same write.

The n-back-3 probe remains at chance. That is the important limitation: the
cell currently transfers a learned working-memory procedure to new state, but
does not automatically acquire a longer rule from the n-back-2 foundation.
The next task is a protected-prefix rule-growth experiment in which a new
external computation is learned for n-back-3 while the n-back-2 cell remains
frozen and causally retained.

Reproduce with:

```bash
PYTHONPATH=src:. uv run python -m experiments.brainworkshop_canonical.causal_working_memory_transfer \
  --seed 17 \
  --report-out /tmp/causal-working-memory-transfer-17.json
```

Reports:

- `report_seed_17.json`
- `report_seed_18.json`
- `sample_efficiency_ledger.json`

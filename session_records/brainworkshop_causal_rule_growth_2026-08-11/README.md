# Causal protected rule growth — 2026-08-11

This audit tests the next step after causal external working-memory transfer:
acquire a new n-back-3 rule in an appended external working-memory cell while
the mastered n-back-2 cell and the controller remain frozen. The two rules are
selected by ordinary rendered cue symbols (`4` and `5`), which enter the
system as learned stimulus events rather than task metadata.

| seed | old retention before | old retention after | new retention | old routed | new routed | shuffled-cue target selection |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.5398 |
| 18 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.5540 |

Both seeds pass every promotion gate: complete-prefix retention, target
mastery, unchanged controller and source codec digests, exact cue-conditioned
route separation, route-state reload with the compatible learned-event
encoder, non-destructive reversal, and zero replayed training examples. The
training budget was 64 source updates plus 256 target updates per seed; the
audit itself consumed no optimizer updates.

This promotes a bounded architectural capability: a frozen controller can
grow a new causal working-memory file and route it from an opaque learned
event while protecting an earlier file. It does not establish arbitrary rule
induction, unrestricted memory growth, open-ended compression, or general
continual learning. The route table is keyed by a versioned learned-event
representation, so route state must be restored with the compatible encoder
version.

Reproduce one seed with:

```bash
PYTHONPATH=src uv run python -m experiments.brainworkshop_canonical.causal_rule_growth \
  --seed 17 \
  --report-out /tmp/causal-rule-growth-17.json
```

Reports:

- `report_seed_17.json`
- `report_seed_18.json`
- `sample_efficiency_ledger.json`

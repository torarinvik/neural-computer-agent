# Independent intention memory under nonstationarity — 2026-08-10

This record promotes the next architecture seam after the first
outcome-trained generator rung. `ExternalOutcomeIntentionMemory` separates
controller batch size from external-cell capacity: one opaque controller state
can query every memory cell, and planner candidate provenance identifies the
cell whose emitted intention was actually attempted.

The pressure test masks half of the controller state, delays verifier outcomes,
adds 20% outcome noise during reversal, protects mastered cells, performs
repeated growth, and compacts redundant verified repertoire entries. A copied
cell is first tried for the reversal. When its held-out learning curve is
clearly worse, the candidate transaction is rolled back and a fresh cell is
grown. This is an explicit guard against negative transfer, not silent weight
retention.

| seed | source | successor | inherited reversal probe | fresh reversal | fresh successor | noisy control | shuffled score |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 85201 | 36 / 0.992 | 7 / 0.965 | 60 / 0.000 | 47 / 0.973 | 40 / 0.965 | 35 / 0.929 | 0.004 |
| 85202 | 38 / 0.960 | 8 / 0.966 | 60 / 0.264 | 56 / 0.979 | 52 / 0.963 | 49 / 0.922 | 0.388 |

Both seeds pass the frozen-controller, partial-context, delayed-credit,
copy-on-write retention, reversal rollback, fresh-transfer, noisy, shuffled,
repertoire admission/consolidation, exact reload, and zero-replay gates. Warm
successor learning is `5.71x` and `6.50x` cheaper than matched fresh learning
in update count.

The claim boundary remains bounded: this is nonstationary external intention
cell growth with a transactional negative-transfer safeguard. It is not
general continual learning, autonomous memory routing, unrestricted memory
growth, or arbitrary program induction. The next pressure is learned routing
over many contexts and cells, with no lifecycle-selected cell supplied by the
caller, followed by open-ended consolidation and Brain Workshop transfer.

Reproduce with:

```bash
.venv/bin/python experiments/policy_free_intention_memory/train.py \
  --seed 85201 \
  --report-out /tmp/policy-free-intention-memory-85201.json
```

Reports:

- `report_seed_85201.json`
- `report_seed_85202.json`
- `sample_efficiency_ledger.json`
- `SHA256SUMS`

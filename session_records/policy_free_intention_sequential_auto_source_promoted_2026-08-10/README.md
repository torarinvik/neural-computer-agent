# Automatic verified source retrieval for sequential admission — 2026-08-10

This promotion removes the remaining fixed lifecycle address from the
sequential admission audit. Before each copy-or-fresh challenger, the
external router deterministically selects a source from protected, verified
cells using learned route keys, verified prototypes, observed support,
quarantine state, and mask-profile compatibility. The caller supplies no
physical source index to the admission transaction.

The three unseen families still select `transfer -> fresh -> transfer` across
all warm and matched-fresh runs. The source selector evaluates 8, 9, and 10
verified candidates as memory grows; it chooses cell 7 in most runs and cell 6
for one warm seed. Every admission passes cost-aware v2 selection, mastery,
complete-prefix retention, persistence, frozen-core, causal controls, and
post-reversal retention. External memory grows append-only from 8 to 11 cells.

| seed | warm selected source cells | fresh selected source cells | warm/fresh unique bits |
| ---: | --- | --- | ---: |
| 85301 | `7 / 7 / 7` | `7 / 7 / 7` | `91 / 89` |
| 85302 | `7 / 7 / 7` | `7 / 7 / 7` | `84 / 109` |
| 85303 | `6 / 6 / 6` | `7 / 7 / 7` | `85 / 82` |

The selected source coverage is at least `0.75` in every branch, and all
source-selection receipts are versioned and auditable. Replayed examples are
zero.

Reproduce from the repository root:

```bash
.venv/bin/python experiments/policy_free_intention_routing/sequential_admission.py \
  --seed 85301 \
  --report-out /tmp/policy-free-intention-auto-source.json
```

This promotes bounded automatic source retrieval for sequential external
memory admission. Broad source generalization, arbitrary new computation,
unrestricted growth, compression, and general continual learning remain
unqualified.

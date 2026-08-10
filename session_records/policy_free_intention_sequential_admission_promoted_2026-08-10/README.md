# Sequential cost-aware external-memory admission — 2026-08-10

This audit moves beyond one-shot challenger selection. After the known
adaptive evidence curriculum, three pairwise-distinct unseen task families
are admitted sequentially from the same protected successor file:

1. nearby successor target — transfer expected;
2. unrelated target — fresh expected;
3. alternate nearby successor target — transfer expected.

Each admission creates isolated transfer and fresh candidates, applies a
nonzero cost-aware utility, commits only the selected branch, and then runs a
complete-prefix held-out verifier over every earlier accepted file before the
next admission. The memory grows append-only from eight to eleven cells.

All three seeds select `transfer -> fresh -> transfer` in both warm and
matched-fresh runs. Every task masters and retains, every prefix verifier
passes, all receipts use the v2 cost-aware schema, and the latest capability
survives a later reversal stress test. Missing-evidence, corruption,
shuffled-outcome, exact-persistence, frozen-core, and zero-replay controls
also pass.

| seed | warm task updates | matched-fresh task updates | warm/fresh unique bits |
| ---: | --- | --- | ---: |
| 85301 | `11 / 7 / 1` | `5 / 1 / 11` | `91 / 89` |
| 85302 | `1 / 10 / 1` | `15 / 11 / 11` | `84 / 109` |
| 85303 | `11 / 1 / 1` | `1 / 1 / 8` | `85 / 82` |

Each run spends `96` held-out prefix-verifier bits, `320` causal-control
outcome bits per side, and replays zero examples.

Reproduce from the repository root:

```bash
.venv/bin/python experiments/policy_free_intention_routing/sequential_admission.py \
  --seed 85301 \
  --report-out /tmp/policy-free-intention-sequential-admission.json
```

This promotes bounded sequential cost-aware admission and complete-prefix
retention across three synthetic unseen families. It does not establish broad
distributional generalization, arbitrary new computation, unrestricted
memory growth, compression, or general continual learning.

# Frequency–recency replacement milestone — 2026-07-26

## Result

Capacity-six replacement now composes two generic memory-utility signals:
recency and ordinary retrieval frequency. The promoted controller has 298,359
parameters; only one new zero-initialized residual coefficient was trained.

Two independent 20-update reward-only runs passed:

- seed 6607: 95.32% held-out, 87.30% correct eviction, 3.23 seconds training;
- seed 6608: 95.10% held-out, 86.13% correct eviction, 3.23 seconds training;
- 51,200 unique verifier bits per run, no replay, no utility labels;
- recency, binary mapping, and four-rule retention passed on both.

Two physical audits passed:

- seed 6701: 96.81%, 92.97% correct eviction;
- seed 6702: 96.29%, 93.36% correct eviction;
- all 512 access histories survived save/reload exactly;
- 3,072 rows before and after, zero capacity growth;
- both age and frequency corruption caused material behavioral drops.

Promoted checkpoint:
`artifacts/checkpoints/unified_memory_frequency_recency_capacity6_seed6607.pt`

SHA-256:
`1346da994de4ba20864c5f1bc1da12684fc13d8dcda480a76cfc6f713da0181c`

Independent replica:
`artifacts/checkpoints/unified_memory_frequency_recency_capacity6_seed6608.pt`

SHA-256:
`b50a3338ef197c4cd955b45a465994052df6443772c73c2bd97c421f1440bc8f`

## Failure localization that mattered

1. Widening the saturated inherited MLP did not produce causal frequency use.
2. Exploration temperature moved the new weight but not decisions.
3. The cold exponential reward baseline gave every action positive advantage
   during tiny runs; batch-centered verified advantage fixed the direction.
4. Feeding uncentered frequency shifted all real rows against skip; centered
   log frequency removed that artifact.
5. A direct one-parameter residual preserved the inherited recency path and
   learned the new trade-off in 20 updates.

## Honest boundary

This is a fixed learned frequency–recency mixture. The controller has not yet
shown that it can adapt online when the utility mixture switches, nor that it
can consolidate or merge an unbounded stream.

## Next atom

Use a capacity-six piecewise-stationary stream whose frequency/recency mixture
switches. Begin with one switch and a generous visible history. Measure
verified regret and unique verifier bits to recover after the switch. Only
scale duration after a sub-minute run shows causal adaptation without losing
the prior mixture.

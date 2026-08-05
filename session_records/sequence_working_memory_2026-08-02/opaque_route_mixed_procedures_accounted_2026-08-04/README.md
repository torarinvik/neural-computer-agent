# Replicated opaque-address execution routing

This record tests whether a frozen working-memory controller can emit an
opaque query that a replaceable external memory/router uses to select the
right executable growth artifact.

The parent is a span-ten controller. The two artifacts share one growth
schema: the existing mixed procedure and a separately learned complement
procedure. The router receives no procedure name, task ID, correct-row label,
or raw modality. During training it sees only random opaque row keys,
attempted rows, and the deterministic scalar outcome of each attempt. The
selected tensor artifact is loaded through `load_growth_artifact` and run by
the frozen controller.

## Results

| Seed | Normal | Reward shuffled | Permuted rows | Cosine baseline | Mixed selected / zeroed | Complement selected / wrong / zeroed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64001 | 100% | 50% | 100% | 50% | 88.1% / 81.7% | 74.5% / 51.2% / 30.0% |
| 64002 | 100% | 50% | 100% | 50% | 88.4% / 82.8% | 75.8% / 51.6% / 29.5% |

Both runs passed exact bank reload, frozen-core identity, and corruption
rejection. The mixed wrong-row control was not discriminative because the
other artifact performed equally well on that procedure; it is not counted
as evidence. The complement wrong-row and both zero-artifact controls were
causal.

## Accounting and boundary

Each run generated 2,048 unique logical lifetimes and 64 unique
query/attempt verifier pairs. It sampled 32,768 verifier bits across 512
optimizer updates; 32,704 samples revisited an already observed pair. Router
latency was approximately 0.0014–0.0017 ms per query on the local CPU audit.

This is a replicated diagnostic for opaque address routing into isolated
executable growth. It does not establish general address discovery, arbitrary
program induction, cold-start skill acquisition, or broad continuous
learning. The next pressure test should vary the public cue and artifact
family while making every wrong-address control behaviorally discriminative.

Reports:

- `seed64001/report.json`
- `seed64002/report.json`

The harness is
`experiments/working_memory_continuous/route_mixed_procedures.py`.

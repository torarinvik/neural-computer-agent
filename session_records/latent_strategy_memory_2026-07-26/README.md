# Latent strategy memory — 2026-07-26

## Implemented

A capacity-bounded four-slot latent strategy bank now provides fast RAM state
without adding a second controller.

Each record contains:

- a task-agnostic latent key derived from physical-history statistics and
  recent centered verifier responses;
- a two-value generic utility strategy;
- usage counts;
- verified success/failure counts.

The bank supports content-addressed retrieval, verified-reward updates,
capacity-bounded replacement, and exact save/reload. The normal controller
remains shared. No semantic labels, task IDs, utility weights, or private
verifier answers are exposed.

## Mechanical result

`strategy_memory_mechanical_seed7050_banks4.json` passed:

- capacity remained bounded;
- every strategy save/reload was exact;
- physical and tensor rewards matched within `1e-6`;
- old binary and four-rule skills were retained; and
- only the permitted utility residual affected model state.

## Capability result

The capability gate did not pass.

Seed 7042:

- global target reward-AUC: `-0.01042`
- strategy-memory target reward-AUC: `+0.02083`
- global verifier bits: 2,592
- strategy verifier bits: 3,360
- strategy remained better per 1,000 bits

Seed 7043:

- both arms produced `+0.0625` raw target reward-AUC;
- the global residual was more efficient because strategy retrieval required
  an extra probe.

Shuffling strategy keys did not lower reward on seed 7042. Thus the positive
pilot is not causally attributable to correct context addressing.

Canonical report:
`experiments/archive/unified_cognitive_controller/reports/strategy_memory_audit_seeds7042_7043.json`

## What this localized

The memory machinery is no longer the bottleneck. The fixed context key is.
Physical summary keys were nearly indistinguishable across utility conditions,
and reward signatures improved separability without producing replicated,
key-dependent retrieval.

## Next experiment

Train a very small context encoder through verifier reward:

1. input: physical-history summaries and recent outcome signatures;
2. output: normalized latent strategy key;
3. training signal: which retrieved strategy produces the best verified
   future reward;
4. no task labels or utility metadata;
5. compare learned keys with fixed keys, shuffled keys, cold memory, and the
   global residual;
6. charge every probe and controller operation;
7. require two-seed improvement per verifier bit plus retention.

Start with a linear encoder and a symmetric perturbation horse race. Increase
capacity only if correct-key retrieval becomes causally useful.

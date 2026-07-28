# Conditional per-query memory usage prior

## Question

Can one controller learn from scalar visual reward when exact content should
dominate retrieval and when verified past usage should break an ambiguous
content tie?

## Design

The parent has already selected global scale zero. This experiment adds a
dormant 49-parameter policy that receives only:

1. top cosine similarity;
2. the top-two cosine margin;
3. usage of the content-leading row;
4. occupied-row count.

The output is a Bernoulli choice between scale zero and scale one for each
query. Exact-query banks reward scale zero; near-duplicate banks reward scale
one. Training uses only the controller's attempted retrieval and resulting
scalar visual-task success. The arm identity and preferred scale remain private
audit metadata.

## Pre-registered gates

- at least 90% overall and 88% on each arm;
- beat both fixed scales by at least eight points;
- at least 95% conditional-action accuracy;
- feature shuffle costs at least 20 action-accuracy points;
- value corruption costs at least 15 task-accuracy points;
- at least 88% after physical save/reload with exact persistence;
- binary-mapping and four-rule retention;
- only the conditional policy changes;
- under five minutes.

## Results

| run | stable verifier bits to 95% | held-out | exact | ambiguous | physical |
|---|---:|---:|---:|---:|---:|
| normal 17603 | **5,120** | **100%** | **100%** | **100%** | **100%** |
| normal 17604 | **5,120** | **100%** | **100%** | **100%** | **100%** |
| reward shuffled 17605 | never | 73.63% | 100% | 47.27% | 73.44% |

For seed 17603, fixed scale zero scored 74.22% and fixed scale one 74.41%.
For seed 17604, they scored 73.05% and 74.80%. Both successful policies became
perfect at update 40 and stayed perfect at updates 48, 56, 64, 72, and 80.
Each run used 10,240 unique verifier bits, 20,480 logical contexts, and no
replay.

Feature shuffling reduced conditional-action accuracy to 50.0% and 51.37%,
with task accuracy at 73.83% in both seeds. Corrupting retrieved values reduced
task accuracy to 48.63% and 47.85%. The reward-shuffled run stayed on the
content-first behavior and failed all learning gates.

Each successful run created, saved, and reloaded 256 independent physical
two-row banks. Every bank persisted exactly, and both policies achieved 100%
accuracy on both arms after reload.

## Inherited capability audits

The promoted seed-17603 checkpoint retained:

- selective disk: 94.14% first reload, 92.77% repeat reload, 65.63% under
  wrong-value corruption, and all gates accepted;
- unequal-strength volatility: 100% valid replacement, 98.96% visual
  accuracy, all 128 histories exact after reload, and all gates accepted;
- binary mapping and four-rule behavioral gates.

The full repository suite passed: 407 tests plus 15 subtests.

## Conclusion

This is a verified conditional retrieval breakthrough: scalar sensory reward
trained a tiny generic policy to choose between two incompatible memory-use
strategies per query, outperforming either global strategy by more than 25
points while preserving earlier skills and physical persistence.

The next gradual experiment should replace the binary choice with a continuous
scale and test transfer to three or four competing rows before attempting
naturally occurring duplicate memories.

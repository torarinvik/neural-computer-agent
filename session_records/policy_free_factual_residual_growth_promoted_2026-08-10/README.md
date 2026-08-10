# Promoted: one-pass factual residual growth

Date: 2026-08-10  
Seeds: `101`, `102`  
Schema: `neural-computer.policy-free-factual-residual-growth.v1`

## Result

Both seeds pass the factual-residual promotion gates. A frozen shared
transition model acquires a successor regime through one context-addressed
random-feature residual. The candidate passes an independent held-out
one-step probe, a two-step recursive rollout, and source-retention probe;
shuffled transition evidence is rejected and missing evidence is a no-op.

| seed | residual held-out MSE | rollout MSE | source-retention MSE | shuffled MSE |
| ---: | ---: | ---: | ---: | ---: |
| 101 | 0.000890 | 0.000551 | 0.004217 | 2.160093 |
| 102 | 0.005689 | 0.028136 | 0.000352 | 1.028655 |

The residual consumes `32` unique transition rows once, performs zero residual
optimizer updates, and leaves the shared base byte-stable. Full-model-copy and
fresh controls replay the target rows through `1,500` optimizer updates and
fail source retention at target stability. This is evidence for factual
residual acquisition and retention, not general continual learning,
unrestricted residual capacity, arbitrary program induction, or policy
learning.

Reports: `seed-101.json`, `seed-102.json`.

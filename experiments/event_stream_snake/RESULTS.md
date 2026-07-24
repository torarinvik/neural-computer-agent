# Event-stream Snake results

The first controlled run used three seeds, 4,000 teacher examples, ten supervised
epochs, and 20 closed-loop evaluation episodes per seed. The learned-threshold model
received 20 additional short-segment policy-gradient episodes. SmolVLM2-500M remained
frozen throughout.

| Adapter | Offline action | Apples / episode | Steps / episode | Vision tokens | Audio tokens | Mean latency |
|---|---:|---:|---:|---:|---:|---:|
| dense | 71.1% ± 5.7% | **0.80 ± 0.41** | **13.12** | 12.00 | 12.00 | 13.60 ms |
| fixed event gate | **72.2% ± 2.9%** | 0.75 ± 0.39 | 13.03 | 3.54 | 0.17 | 13.81 ms |
| learned thresholds + RL | 69.7% ± 2.0% | 0.42 ± 0.29 | 8.58 | **3.29** | **0.12** | 13.92 ms |

## What worked

The fixed raw pixel-change and PCM-energy gate removed 84.5% of sensory tokens while
preserving behavior. Its small differences from dense delivery are well inside this
three-seed development run's variation. This supports the core event-stream premise:
unchanged frames and literal silence need not bombard the frozen listener.

The learned thresholds were interpretable and consistent. Vision moved from 0.002 to
0.0027–0.0031; audio moved from 0.020 to 0.044–0.049. The resulting stream removed
85.8% of dense tokens.

## What did not work

The learned threshold policy did not improve task reward. Policy-gradient tuning moved
mean offline accuracy from 70.8% to 69.7% and reduced apples relative to both controls.
Sparse immediate rewards and short credit assignment encouraged stricter emission
without teaching which events improve long-horizon survival.

Token reduction also did not reduce measured latency. At only 24 dense sensory tokens,
the fixed cost of the 500M transformer and adapter dominates attention work; a
4-versus-24-token sequence is too small to expose an attention-scaling benefit.

Every controller died in every episode. The result is therefore a successful
event-filtering/interface-efficiency proof and a negative learned-reward result—not a
successful Snake agent.

Machine-readable aggregates are in `results.json`; raw run JSON is archived in
`raw_results_20260716.tar.gz`.

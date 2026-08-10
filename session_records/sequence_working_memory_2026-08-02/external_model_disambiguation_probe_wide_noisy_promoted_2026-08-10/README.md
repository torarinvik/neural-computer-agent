# Wide noisy factual-model disambiguation probe — promoted

This three-seed follow-up widens the candidate intention space from two to
eight and adds Gaussian outcome noise with standard deviation `0.1`. Only the
last intention is diagnostic; the active planner must discover it from
factual prediction disagreement. A uniform random-intention and random-tie
control remains the comparison floor.

| metric | seed 83101 | seed 83102 | seed 83103 |
| --- | ---: | ---: | ---: |
| active-probe routing accuracy | 0.984 | 0.977 | 0.980 |
| random-control routing accuracy | 0.801 | 0.820 | 0.754 |
| causal probe margin | 0.184 | 0.156 | 0.227 |
| candidate intentions | 8 | 8 | 8 |
| controller updates | 0 | 0 | 0 |
| raw replayed examples | 0 | 0 | 0 |
| exact persistence | true | true | true |

All seeds pass the predeclared noisy quality gate: active routing at least
`0.95`, active selection of the informative intention, and superiority to the
random control. This strengthens the narrow probe boundary but does not yet
integrate probe requests into a live asynchronous router or establish learned
probe selection, multimodal usefulness, or general continual learning.

Reports are protected by `SHA256SUMS`.

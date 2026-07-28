# Continuous per-query memory usage prior

## Question

Can the controller reuse its learned binary retrieval skill to acquire a more
efficient continuous policy with very little new verified experience?

## Benchmark correction

The first pilot was rejected. Its exact and ambiguous valid-scale intervals
overlapped, allowing one constant scale to solve both. Its physical audit also
treated the memory reader's returned confidence as a row index.

The promoted benchmark fixes both problems:

- exact queries succeed only below a boundary sampled from `0.12–0.18`;
- ambiguous queries succeed only above a boundary sampled from `0.35–0.55`;
- fixed scale zero and fixed scale one each score exactly 50% row accuracy;
- physical correctness is checked against the opaque retrieved value.

The inherited policy already distinguishes the query regimes. Training samples
a continuous scale, observes only visual success, and pays a smaller generic
penalty proportional to its own scale. Correctness remains more valuable than
the maximum possible resource saving.

## Pre-registered gates

- at least 95% two-row accuracy;
- reduce mean scale by at least 0.15 from the inherited binary value `0.50`;
- improve verified correctness-minus-cost utility by at least 0.01;
- at least 93% zero-shot four-row accuracy;
- at least 20 points lost under feature shuffling;
- at least 15 visual-accuracy points lost under value corruption;
- at least 93% after physical save/reload at every row count;
- exact persistence and all inherited retention gates;
- only the existing conditional policy changes;
- under five minutes.

## Results

| run | stable bits | held-out rows | mean scale | 4-row zero-shot | accepted |
|---|---:|---:|---:|---:|---:|
| normal 17718 | **640** | **100%** | **0.312** | **100%** | yes |
| normal 17719 | **640** | **100%** | **0.347** | **100%** | yes |
| reward shuffled 17720 | never | 50% | 0.121 | 50% | no |
| reset policy 17721 | never | 50% | 0.091 | 50% | no |

Each normal run used eight updates, 2,048 unique logical contexts, 1,024
verifier bits, and no replay. Training took 5.17 and 5.23 seconds. Both first
crossed the joint correctness-and-efficiency gate at update five and remained
above it at updates six, seven, and eight.

The inherited binary policy was already correct but used mean scale `0.50`.
The descendants retained 100% accuracy while reducing that resource by 37.6%
and 30.6%. Their verified utility rose from `0.85` to `0.906` and `0.896`.

## Causal and physical audits

- feature shuffle: four-row accuracy fell to 53.32% and 49.41%;
- value corruption: visual success fell to 45.70% and 48.93%;
- reward shuffle: the policy collapsed to the exact arm, 50%;
- reset policy: the same architecture and experience remained at 50%;
- original binary conditional retrieval: 100% in both descendants;
- physical two/three/four-row retrieval: 100% across 128 banks per size and
  per seed, with every bank reloaded exactly.

The selected seed-17718 checkpoint also retained:

- selective disk: 92.58% first reload, 92.97% repeat reload, 66.41% with
  wrong values, and every gate accepted;
- unequal-strength volatility: 100% valid replacement, 99.74% visual
  accuracy, all 128 histories exact and bounded, and every gate accepted;
- binary mapping and four-rule behavioral gates.

## Conclusion

This is a compounding sample-efficiency result. The previous discrete skill
made a continuous resource-control improvement learnable in 640 new verifier
bits. The identical reset learner could not acquire it in that budget, while
shuffled feedback destroyed it.

The next gradual frontier should make a third row the correct target on some
queries and broaden the boundary distribution. Natural duplicate memories
come only after that controlled three-way relation passes the same audits.

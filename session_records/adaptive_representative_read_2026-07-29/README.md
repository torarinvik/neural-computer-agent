# Action-conditioned adaptive representative reading

## Question

Can the controller predict when consulting additional representatives will
improve its answer, preserve essentially all of the accuracy of a six-row
read, and pay the extra comparison cost only on difficult events?

Accuracy remains sovereign. Each latent comparison carries a small secondary
cost of `0.00025`.

## Localization before repair

A fixed one-representative read already succeeds on roughly 98–99% of events.
An ordinary success predictor therefore collapsed to “the cheap action will
work” at every tested budget through 12,288 verifier outcomes.

A corrected diagnostic asked the causal question: from the fresh latent and
the first representative of each learned equivalence class, is the event on
which deeper reading improves the answer decodable? A 64/16-unit MLP reached
`0.9941` held-out AUC:

- the top-scored 1% of events contained 83.3% true improvement cases;
- the top 2% captured 92.0% of all improvement cases;
- the signal was finite and held across bars, diamonds, and dot pairs.

The information was present. The missing piece was learning from rare
marginal improvements rather than from overwhelmingly positive absolute
success.

## Architecture and learning signal

The controller gains a 32,097-parameter critic. Its inputs are entirely
latent:

- the fresh feedback-derived memory value;
- the first stored representative from each learned cluster;
- their absolute difference and elementwise product;
- the two relation scores and validity bits.

The two possible actions are:

- shallow: consult the first representative in each class;
- deep: consult up to three representatives in each class.

Both actions are executed during training. The target is derived only from
their scalar outcomes: did deep reading succeed when shallow reading failed?
A class-balanced loss makes that rare verifier event learnable. The learner
never receives an appearance name, rule bit, correct row, or correct compute
budget.

The 0.01 decision threshold and critic initialization were selected in a
12-candidate horse race. All candidates reused the same verifier data.
Architecture search consumed 147,456 unique verifier outcomes and is
accounted separately from deployed learning.

## Sample-efficiency boundary

| formal budget | seed | adaptive utility beats deep | accepted |
|---|---:|---:|---:|
| 4,098 examples / 8,196 bits | 20711 | no, by 0.000084 | no |
| 4,098 examples / 8,196 bits | 20712 | yes | yes |
| **8,196 examples / 16,392 bits** | **20721** | **yes** | **yes** |
| **8,196 examples / 16,392 bits** | **20722** | **yes** | **yes** |

The smaller budget is a best case, not a stable frontier. The promoted
replicated budget is 16,392 verifier outcomes and 1,000 optimizer updates.

## Formal held-out results

Each replica is evaluated on 49,152 disjoint events.

| metric | seed 20721 | seed 20722 |
|---|---:|---:|
| adaptive accuracy | **99.573%** | **99.567%** |
| always-deep accuracy | 99.565% | 99.618% |
| shallow accuracy | 98.665% | 98.539% |
| adaptive comparisons | **2.092** | **2.094** |
| always-deep comparisons | 5.996 | 5.997 |
| comparison reduction | **65.1%** | **65.1%** |
| adaptive verified utility | **0.995205** | **0.995143** |
| always-deep verified utility | 0.994147 | 0.994676 |
| deep-read rate | 1.34% | 1.49% |

The critic spends more on the hardest appearance without being told its name.
For seed 20721 it deep-reads 0.60% of bars, 0.96% of diamonds, and 2.45% of
dot pairs. In the first replica it is also slightly more accurate than
always-deep reading because extra representatives occasionally mislead; the
critic learns to avoid those cases.

## Causal and physical audits

- Reversed-rule adaptive accuracy is 99.58% and 99.53%.
- Shuffling critic features across events reduces accuracy to 98.69% and
  98.56%, close to the shallow policy.
- Zeroing memory values reduces accuracy to 49.27% and 50.29%.
- Shuffling verifier outcomes during training yields 98.70% and 98.54%,
  fails the deep-utility, reverse, physical, and feature-causality gates, and
  is not admitted.
- All 1,024 physical six-row banks reload exactly.
- Physical adaptive accuracy is 99.35% and 99.28%, within the pre-registered
  half-point allowance of physical always-deep accuracy.
- Binary-mapping and four-rule retention gates pass.
- Every inherited tensor remains bit-identical.
- Reloading the promoted checkpoint reproduces every held-out metric exactly.

## Promoted artifact

`artifacts/checkpoints/unified_adaptive_representative_read_seed20721.pt`

Total controller parameters: 343,398.

SHA-256:
`6e23eea859b9ccbfa3f3fa28b828de20520ae216ef376c7b899dbb5c063c256e`

## Conclusion and next frontier

The controller now uses an action-conditioned prediction of marginal success
to allocate memory-read compute. It preserves essentially full six-row
accuracy while executing only about one third as many comparisons, and it
learns the allocation from verified outcomes rather than a hand-written
difficulty label.

This solves adaptive read depth, not adaptive storage. The six
representatives still occupy memory even when only two are consulted. The
next frontier is delayed physical pruning: accumulate evidence about whether
the extra representatives ever change verified outcomes, then delete them
only when predicted future value falls below their storage and reload cost.

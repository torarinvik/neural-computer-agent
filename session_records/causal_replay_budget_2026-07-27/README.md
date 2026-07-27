# Causal replay-budget breakthrough — 2026-07-27

## Question

Can the controller reduce verifier experience by spending more internal
processing on each already-observed experience, and can a learned stopper safely
remove apparently unhelpful processing?

All comparisons use paired-population feedback, the six-action composition
frontier, identical seeds, 64 experience batches, and 60 lifetimes per batch.

## Learned-stopper localization

Rich decision-time learner diagnostics made future behavioral improvement
rank-predictable across the five-to-six-action boundary:

- linear held-out correlation: `0.310`;
- longer 16-batch target correlation: `0.342–0.346`;
- the predicted bottom quartile was harmful on held-out fixed trajectories;
- shuffled targets erased the separation.

This observational result did **not** survive intervention. Recent-window
pruning saved `8.6–9.4%` of optimizer updates but delayed stable mastery and
reduced final utility. A late-only policy was safer but failed replication.
The reason is now localized: evaluating a block while holding the later
trajectory fixed does not estimate the long-run causal consequence of omitting
that block, because every later update then starts from a different state.

The stopper is retained as a diagnostic, not promoted as the default policy.

## Exact causal budget comparison

We therefore compared complete matched trajectories with fixed budgets. Eight
updates was worse than sixteen on all eight seeds, losing `0.0185–0.0381`
final utility and usually failing stable mastery. Twenty-four updates then beat
sixteen on every seed:

| Seed | Stable bits @16 | Stable bits @24 | Experience reduction | Final utility gain | Updates to mastery @16/@24 |
|---:|---:|---:|---:|---:|---:|
| 8220 | 6,240 | 5,040 | 19.2% | +0.01629 | 416 / 504 |
| 8221 | 3,360 | 2,400 | 28.6% | +0.00337 | 224 / 240 |
| 8222 | 6,480 | 4,320 | 33.3% | +0.00896 | 432 / 432 |
| 8223 | not reached | 5,040 | rescued | +0.01708 | — / 504 |
| 8224 | 6,720 | 5,280 | 21.4% | +0.01200 | 448 / 528 |
| 8225 | 5,760 | 4,080 | 29.2% | +0.01234 | 384 / 408 |
| 8226 | 6,480 | 3,600 | 44.4% | +0.02936 | 432 / 360 |
| 8227 | 6,480 | 5,040 | 22.2% | +0.01453 | 432 / 504 |

## Extended causal ladder

The fixed ladder kept improving well beyond 24 updates:

| Comparison | Matched streams | Median stable bits | Verifier-bit wins | Utility wins | Mean utility change |
|---|---:|---:|---:|---:|---:|
| 32 vs. 24 | 8 | 3,240 vs. 4,680 | 7/8 | 6/8 | +0.00319 |
| 40 vs. 32 | 8 | 2,640 vs. 3,240 | 7/8 | 6/8 | +0.00319 |
| 48 vs. 40 | 12 | 2,280 vs. 2,880 | 9/12 | 9/12 | +0.00364 |
| 56 vs. 48 | 12 | 2,040 vs. 2,280 | 9/12 | 4/12 | -0.00140 |

Forty-eight is therefore the current verified sweet spot for this six-action
family: it substantially reduces scarce verifier experience while retaining a
positive utility trend. Fifty-six is an overthinking regime: it usually reduces
the bits to mastery, but its final capability regresses often enough to fail the
accuracy-first objective.

An independently trained 48-update checkpoint also passed the full six-action
audit ladder: independent confirmation, feature-shuffle and reversal causality,
exact reload, binary and four-rule retention, persistent skill commit, and
corruption detection.

The next frontier is now causal compute allocation. The controller should learn
to choose between safe budgets around the sweet spot only from complete matched
trajectory outcomes—not from local loss reduction—because omission changes all
later learning dynamics.

Raw reports are in [`raw/`](raw/). Tiny diagnostic checkpoints remain local
artifacts and are intentionally excluded by the repository's weight policy.

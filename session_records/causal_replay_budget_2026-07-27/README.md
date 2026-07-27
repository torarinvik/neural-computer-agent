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

## Conclusion

The next processing frontier is not “stop 16-update replay early.” Sixteen is
often under-compute. Spending 50% more internal compute reduced scarce verifier
experience by 19–44% on every jointly solved stream and rescued one failure.
This is a verified sample-efficiency gain, though not always an optimizer-step
gain: five seeds paid modestly more internal updates to reach mastery, one tied,
and one used fewer.

The next bracket should test 24 versus 32 before attempting another learned
budget policy. A learned controller is only justified if the causally optimal
budget varies across streams; otherwise the evidence supports a better fixed
budget.

Raw reports are in [`raw/`](raw/). Tiny diagnostic checkpoints remain local
artifacts and are intentionally excluded by the repository's weight policy.

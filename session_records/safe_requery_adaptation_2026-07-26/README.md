# Safe re-query adaptation — pre-registration

## Hypothesis

The robust re-query policy should remain deployed while a separate challenger
learns. The challenger may replace it only when randomized attempted-action
experience provides statistically conservative evidence that the challenger
has higher verified utility.

## Learner-visible evidence

For each fresh context the logger randomly attempts ordinary read or re-query
with propensity `0.5`. The system receives only:

- the four generic memory evidence values;
- the attempted operation;
- that operation's scalar verifier outcome;
- the known operation cost and logging propensity.

The unattempted outcome, correct compute choice, correct answer, task identity,
and private evaluation metrics are excluded from training and promotion.

## Promotion rule

The deployed incumbent is frozen. A challenger copy trains from attempted
outcomes. At fixed intervals, both policies are evaluated off-policy on the
same accumulated logged records using inverse-propensity weighting. Promotion
requires the lower 95% confidence bound of the challenger's paired estimated
utility improvement to exceed zero. A rejected challenger cannot alter the
incumbent.

## Sub-minute fork

Run two arms on capacity-six re-query:

1. **mastered incumbent:** the robust capacity-five re-query checkpoint;
2. **genuine-gap incumbent:** a reset head, with an independently learning
   challenger initialized identically.

The mechanism passes only if the mastered incumbent remains within the robust
mastery gates without a harmful promotion, while the genuine-gap arm promotes
only with positive confidence-bounded attempted-outcome evidence and improves
private audited utility. A naive continuously updated copy is retained as the
destructive-update control. No longer run is authorized unless this discovery
seed passes.

## Seed 7951 result and one-axis correction

The conservative mechanism protected the mastered incumbent exactly: no
challenger was promoted, incumbent utility remained `0.6702`, and naive
updating degraded it to `0.6590`. The gap challenger improved private audited
utility from `0.5652` to `0.6462`, but three independent 240-record windows
each had a lower confidence bound below zero, so the system correctly refused
promotion.

The harness had discarded valid evidence after every rejected proposal. Seed
7952 changes only this statistical detail: logged records accumulate while the
incumbent is unchanged and reset only after a real promotion. This remains
valid off-policy evidence because the randomized action propensity is fixed
and the incumbent comparator is unchanged. All learner-visible information,
training, thresholds, and budgets remain identical.

## Seed 7952 result and robustness audit

The corrected mechanism achieved both substantive objectives:

- mastered arm: all challenger proposals rejected; incumbent retained `71.7%`
  accuracy and unchanged utility;
- gap arm: promoted at 240 verifier bits only after the attempted-outcome IPS
  lower 95% confidence bound reached `+0.0531`; incumbent utility improved by
  `2.92` points.

The discovery run is still recorded as a formal near-pass, not a pass, because
the naive control degraded mastered utility by `0.25` points rather than its
pre-registered `0.50`-point minimum. That auxiliary control is stochastic and
is not part of the safe mechanism itself.

Seeds 7953 and 7954 are pre-registered as unchanged robustness audits, not as a
retroactive reclassification of seed 7952. Each must (a) reject every harmful
promotion to the mastered incumbent, preserving its accuracy and utility, and
(b) promote the gap challenger only with a positive confidence lower bound and
improve its audited utility by at least `0.02`. The naive arm remains reported
but does not gate this separate robustness claim.

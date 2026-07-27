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

## First causal schedule control (audit pending)

Before fitting another allocator, we tested the smallest causal time-schedule
hypothesis: use the sample-efficient but occasionally overthinking `56`-update
budget for the first 16 experience batches, then switch to the robust `48`
budget. This is a fixed, verifier-blind schedule; it is not a learned policy.

The two pilot seeds selected the schedule and four fresh seeds confirmed it:

| Comparison | Fresh paired seeds | Mean stable verifier bits | Mean final utility | Replay updates |
|---|---:|---:|---:|---:|
| Fixed 48 | 4 | 1,110 | 0.88950 | 3,072 |
| 56 for batches 1–16, then 48 | 4 | 960 | 0.89086 | 3,200 |

That is a `13.5%` reduction in verifier experience on the four fresh seeds,
with a small mean utility increase. The schedule won verifier efficiency on
three of four fresh seeds; its one loss was 120 bits. Across all six paired
seeds (including the two selection pilots), it averaged 1,360 stable bits
versus 1,660 for fixed 48, while staying within the pre-existing utility
tolerance. The reverse control (48 early, 56 late) used still more internal
updates and had two material utility regressions, so it is not promoted.

This is evidence that *when* internal replay is spent matters, not just its
total amount. A separately trained scheduled checkpoint (seed `8278`) passed
the full six-action audit ladder: independent paired confirmation lower-95
bound `+0.04208`, mean gain `+0.06322`, every fresh stream improved, feature
shuffle and action-reversal both collapsed performance, exact checkpoint reload,
binary and four-rule retention, and persistent-store corruption detection. The
result therefore graduates from a curve-only lead to a verified causal schedule
breakthrough. The raw training report and audit are
`early56_late48_s16_8278.json` and `early56_late48_s16_audit_8279.json`.

An early-state linear allocator was also tested against full matched
trajectories. Its apparent single-split success did not survive four disjoint
seed folds (held-out accuracy `0.17`, `0.67`, `0.67`, and `1.00`), so it is
rejected rather than scaled. The next learned-controller experiment must use
more counterfactual trajectory data, not another local-loss proxy.

## Counterfactual branch allocator — methodology result, not promoted

We implemented the required counterfactual data mechanism. At a decision point
it clones the entire causal learner state—router weights, optimizer moments,
replay buffer, and RNG state—then feeds low- and high-replay clones the same
future sensory/reward episodes. The allocation probe is allowed to see only a
pre-branch internal summary; its label is produced afterward by the verifier.
Whole logical streams, never branch rows, are split between train and test.

The first stationary-stream datasets did not support a learned allocator. Both
the small replay-statistic summary and a richer latent summary failed their
held-out/shuffle gates. This is an honest negative: in an i.i.d. stream, the
unseen future can be random conditional on current state, so a local policy
should not be expected to predict its value.

During a true multi-support extension we caught and fixed two confounds before
using its results:

1. The old helper accepted multiple support trials but only fed one reward into
   the controller. It now consumes every support action and feedback event,
   followed by a dedicated query frame; a unit test checks the exact event
   sequence.
2. An old binary label treated `neither branch reaches stable mastery` as
   `choose low compute`. This could train an allocator to prefer cheap failure.
   Such pairs are now marked ineligible and rejected by the probe loader.

The resulting interpretation is clear. Multiple-support rungs are currently
an unmastered temporal generalization problem for this controller, not evidence
that low replay is preferable. The fixed `56 → 48` schedule remains the only
promoted compute policy. The next high-ROI experiment is a gradual
one-support → two-support → three-support curriculum with rehearsal and
retention gates; only after each rung is reliably mastered should its branch
data be eligible for compute allocation.

Raw reports are in [`raw/`](raw/). Tiny diagnostic checkpoints remain local
artifacts and are intentionally excluded by the repository's weight policy.

## Context-selection frontier — 2026-07-27

The first proposed two-support binary rung was correctly rejected as a false
progress signal: one binary outcome already identifies the whole mapping, and
the parent had previously acquired the four-rule task. Passing it would not
demonstrate a new skill.

The new verifier-only `contextual_mapping` task has two independent hidden
binary mappings selected by an RGB context token; its two supports cover
distinct contexts. The learner still receives only pixels, opaque actions, and
scalar outcomes. The gate requires mastery on both query contexts and a loss
when the second support outcome is removed.

Small probes localize the failure precisely: the frozen visual encoder decodes
context at 100% held-out accuracy; an MLP decodes the two-bit rule from the
post-feedback recurrent state at 97.66% (shuffled labels 20.51%); and a frozen
state-plus-event MLP decodes the correct action at 97.85%. Perception and
state storage are therefore sufficient; outcome-driven action routing is not.

An optional zero-initialized generic prior-state/query-event relation residual
was verified bit-identical at insertion. It, joint reward-only training, and
an attempted-action-only value probe all failed to master the second context
at the sub-minute budget. The value probe learned a causal first-context policy
but reached only 50–53% on the second context, so nothing is promoted.

The next deliberately easier rung is `contextual_override`: one hidden binary
mapping plus one context-specific invariant response. It teaches context
selection before two independent context-bound rules. Its first joint pilot
reached 78.91% held-out overall but failed both-context and reversal gates; no
longer run is justified yet. The rule is now explicit: scale the two-rule task
only after a causal, retained gain on the override rung. Raw reports use the
`contextual_*_837*` prefix.

## Replicated two-skill integration breakthrough — 2026-07-27

The previous apparent conflict between visible-context acquisition and binary
few-shot reasoning was resolved by a two-stage, behavior-only consolidation
curriculum. First, a zero-initialized generic action residual learned static
sensory routing from the independently acquired visible-context specialist
while matching the mature binary controller on old frames. Second, the same
residual underwent 32 updates of complete sensory/action/outcome trajectory
rehearsal against both learned controllers. The target was only their opaque
action distributions; no semantic IDs, verifier rule labels, or correct
unattempted actions were targets.

Seed `8395` and the independent full-pipeline replica `8397` both passed every
registered gate for **both** skills. The replica reached 97.42% held-out binary
few-shot accuracy (98.01% under reversed rules) and 95.54% visible-context
accuracy (94.79% under its pixel counterfactual), with blank context vision at
48.63% and 88.40% action flips. Binary feedback shuffle, state reset, visual
ablation, and reversal controls all remained causal. A shuffled-specialist
distillation control was rejected.

This is the first evidence in this branch that independently acquired abilities
can be retained and merged into a single controller without direct task labels
or catastrophic loss of the earlier skill. The promoted reproducible artifact
is `artifacts/checkpoints/unified_binary_context_integrated_seed8397.pt`,
SHA-256 `332fb1d2d51eea210ac695e101b64fa53ef8ec5059cf1f5bd26755a297089d9c`.

### Visible-context acquisition and integration boundary

A fresh controller *can* acquire the RGB context-to-opaque-action primitive
efficiently. Fresh seeds `8382` and `8383` both reached 100% normal and
counterfactual held-out accuracy after 64 updates (8,192 attempted outcomes),
with blank vision at chance and 100% prediction flips. The replicated seed
`8383` checkpoint is `artifacts/checkpoints/unified_visible_context_seed8383.pt`.

The mature inherited controller did not acquire the same primitive within 64
updates (55.63% held-out). This is direct evidence of selective negative
transfer in the present monolithic update path, not a rendering or task-data
failure. A matched 16-update transfer test then found no positive forward
transfer from the fresh context specialist to `contextual_override`: it reached
69.78% versus 73.24% from a fresh start, and both failed the causal gate.

Therefore the visible-context checkpoint is a replicated *isolated primitive*,
not a general-controller promotion. The next integration experiment must
preserve independently acquired useful representations while testing whether a
single controller can combine them, rather than assuming ordinary joint
fine-tuning will do so.

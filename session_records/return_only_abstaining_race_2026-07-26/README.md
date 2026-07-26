# Prospective return-only abstaining race

Date: 2026-07-26

## Pre-registered protocol

Use unseen physical stream 7083 and clones 7190–7193. Preserve the successful
learner configuration but remove the round-18 shadow audit and all early
training elimination.

1. Train every clone through round 42: eighteen old-equal rounds, eighteen
   reliability rounds, and six genuine old-return rounds.
2. Rank by mean learned-minus-frozen verifier reward over the six return
   rounds, then worst-round advantage, then lowest clone ID.
3. Continue the winner only if mean advantage is greater than `1e-6` and
   worst-round advantage is at least `-1e-6`; otherwise abstain.
4. Continue an earned winner for exactly twelve more rounds.

No shadow outcome, accuracy, target label, semantic ID, or diagnostic enters
allocation. The production path uses 180 rather than 216 rounds when selecting
(16.7% saved), or 168 rounds when abstaining (22.2% saved).

## Validation

For this first prospective test only, complete all stopped clones after the
selection decision. These post-gate diagnostics cannot alter the selected
winner and are excluded from production accounting. They test whether the
six-return-round score actually identifies the best eventual valid learner or
prematurely misses another phase transition.

## Gate

Promotion requires exact resumed prefixes, inherited binary/four-rule
retention, all physical/tensor/persistence gates, and:

- if selecting, the chosen clone must remain the best valid completed learner
  under both reliability and return;
- if abstaining, no completed clone may reveal a valid delayed
  acquisition-and-return trajectory.

## Results

All four clones reached the genuine return screen without shadow-audit
selection cost:

| Clone | Six-round return reward advantage | Worst round | Decision |
|---:|---:|---:|---|
| 7190 | 0.000 points | 0.000 | stop |
| 7191 | **+7.639 points** | 0.000 | continue |
| 7192 | 0.000 points | 0.000 | stop |
| 7193 | +3.472 points | 0.000 | stop |

Clone 7191 earned continuation. Every resumed prefix was exact and every clone
retained both inherited primitives.

The completed diagnostic comparison was:

| Clone | Reliability target advantage | Return target advantage | Reliability reward advantage | Return reward advantage |
|---:|---:|---:|---:|---:|
| 7190 | 0.00 points | 0.00 points | 0.000 points | 0.000 points |
| 7191 | +25.00 points | **+69.44 points** | +3.935 points | **+6.944 points** |
| 7192 | +1.39 points | 0.00 points | +0.231 points | 0.000 points |
| 7193 | **+44.44 points** | +40.28 points | **+6.019 points** | +3.704 points |

## Verdict

The return screen found the best return learner, but the pre-registered
“best under both reliability and return” gate is not satisfied literally.
Clone 7191 dominates return, while 7193 dominates reliability. Neither clone
dominates the other.

This is a useful rejection of return-only ranking for a compounding learner:
retention alone is too narrow, just as acquisition-only round-18 ranking was
too narrow. The allocation objective must explicitly value both learning the
new context and preserving/reusing it on return.

The smallest task-agnostic correction uses only already observed verifier
rewards at round 42:

```text
balanced_score =
    min(
        mean reliability-phase learned-minus-frozen reward,
        mean first-six-return learned-minus-frozen reward
    )
```

Require both components to be positive and the worst return round to be
non-negative; otherwise abstain. On this stream, balanced scores are 0 for
7190, +3.935 points for 7191, 0 for 7192, and +3.472 for 7193. It still selects
7191, but now for the pre-declared reason that it has the strongest conservative
gain across both phases—not merely the largest return score.

Test the maximin selector prospectively on an unseen stream. No early shadow
pruning should return.

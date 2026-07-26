# Successive-halving replication with non-overlapping control

Date: 2026-07-26

## Pre-registered correction

Run the unchanged ladder on unseen physical stream 7079 with clones 7150–7153.
The acquisition and retention rankings remain exactly as previously specified.

The validation control is now the highest-ranked clone eliminated at round 18.
It is determined only by the frozen acquisition ranking and therefore:

- cannot be the selected winner;
- is the strongest trajectory just below the pruning boundary;
- does not use any later reliability or return outcome.

Advance the top two acquisition clones to round 42, advance the retention
winner to round 54, and independently resume the validation control from round
18 to round 54.

## Gate

The replication passes if the winner exceeds this strongest early-pruned
control in the direction of both reliability acquisition and old-return
performance while all inherited-retention, parity, persistence, and exact
continuation gates pass. If either comparison is tied or negative, do not
scale the population mechanism.

## Results

The round-18 acquisition ranking was:

| Rank | Clone | Shadow advantage | Decision |
|---:|---:|---:|---|
| 1 | 7152 | +3.125 points | advance |
| 2 | 7153 | +3.125 points | advance |
| 3 | 7151 | +2.083 points | strongest-pruned control |
| 4 | 7150 | 0.000 points | stop |

At round 42, the six-return-round selector chose 7152:

- 7152 mean reward advantage: +1.389 points, worst 0;
- 7153 mean reward advantage: 0, worst 0.

The completed comparison was:

| Measurement | Selected 7152 | Strongest-pruned 7151 |
|---|---:|---:|
| Reliability target advantage | **+11.11 points** | 0.00 points |
| Old-return target advantage | +20.83 points | **+76.39 points** |
| Old-return reward advantage | +2.546 points | **+7.870 points** |
| Binary retention | pass | pass |
| Four-rule retention | pass | pass |
| Full gate | pass | pass |

The replication gate failed: the selected clone won acquisition but discarded
clone 7151 was a much stronger delayed return learner.

## Localization

This is not a failure of the six-round retention selector. Had 7151 reached
round 42, its first six return rounds already averaged +7.639 reward points
and +75.0 target points, versus +1.389 and +16.67 for 7152. The retention rung
would have selected it decisively.

The failure is specifically over-aggressive pruning at round 18. The
acquisition score distinguishes immediate context switching but cannot rule
out a slightly lower-ranked trajectory that later ignites during return.

## Verdict

Do not scale the two-of-four acquisition prune. It saves 38.9% compute but can
discard a high-value sleeper trajectory.

The smallest evidence-directed repair is to advance three of four clones at
round 18, then use the already successful six-return-round selector at round
42. That would have preserved 7151 and still use only 156 versus 216 physical
rounds, saving 27.8%. Test that exact 3-of-4 ladder prospectively on an unseen
stream before considering larger populations or a new selector.

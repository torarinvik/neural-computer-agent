# Abstaining population-race replication

Date: 2026-07-26

## Frozen replication

Repeat the stream-7081 protocol without modification on unseen physical stream
7082 and clones 7180–7183:

- advance three clones by the round-18 shadow acquisition ranking;
- complete the sole eliminated clone as a blinded validation control;
- rank survivors after six genuine return rounds;
- continue only when the best mean reward advantage is greater than `1e-6`
  and its worst-round advantage is at least `-1e-6`;
- otherwise abstain.

No hyperparameter, ranking, threshold, or tie-break changes.

## Gate

If continued, the selected clone must beat the eliminated control in the
direction of reliability and return and pass all retention/exactness gates.
If abstaining, the eliminated control must also lack a valid acquisition and
return gain. Either outcome must preserve exact resumed prefixes and separate
selection from training accounting.

## Results

The round-18 ranking advanced 7180 (+2.083 points), 7183 (specialization
tie-break), and 7181 (lowest-ID tie-break). Zero-score clone 7182 was
eliminated and independently completed as control.

At round 42, all three survivors had exactly zero learned-minus-frozen reward
advantage on all six return rounds. The frozen rule therefore abstained and
spent no production compute on rounds 43–54.

The abstention was not validated. Eliminated control 7182 later achieved:

- reliability target advantage +4.17 points;
- old-return target advantage +4.17 points;
- old-return reward advantage +0.463 points;
- a passing full gate with both inherited primitives retained.

Its first six return rounds already averaged +1.389 reward points with no
harmful round. Had it survived to round 42, the retention selector would have
continued it.

## Verdict

Rejected. Even three-of-four pruning at round 18 can discard a delayed learner.
The abstention rule itself behaved exactly as specified; the unsafe component
is any training prune before genuine return evidence exists.

The next minimal design is:

1. remove round-18 training elimination entirely;
2. train all four clones through round 42;
3. use the same six-return-round positive/non-harmful threshold to select one
   winner or abstain;
4. continue only that winner for twelve rounds.

This costs 180 rather than 216 rounds when selecting (16.7% saved), or 168
rounds when abstaining (22.2% saved). It also removes the four expensive shadow
audits from the production selector: 448 held-out logical lifetimes and 3,840
verifier bits no longer influence training allocation.

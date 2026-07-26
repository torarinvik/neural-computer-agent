# Balanced maximin population-race replication

Date: 2026-07-26

## Frozen protocol

Repeat the stream-7084 experiment without modification on unseen physical
stream 7085 with clones 7210–7213:

1. train every clone through round 42 with no shadow audit or early prune;
2. compute acquisition gain from mean reliability-phase
   learned-minus-frozen verifier reward;
3. compute return gain from the first six old-return rounds;
4. rank by `min(acquisition_gain, return_gain)`, then return gain, acquisition
   gain, and lowest clone ID;
5. continue only when both gains exceed `1e-6` and every return round is at
   least `-1e-6`; otherwise abstain;
6. give an earned winner exactly twelve more rounds.

No accuracy, target label, semantic identifier, hidden state, or diagnostic
enters allocation.

## Validation gate

Complete all stopped clones after the frozen decision for this replication
audit. The selected clone must retain the greatest final conservative verifier
gain across reliability and return among valid completed clones. If the race
abstains, no completed clone may have positive gain on both axes.

Production accounting remains 180 rounds when selecting or 168 when
abstaining; post-gate diagnostic completions are reported separately.

## Results

At round 42:

| Clone | Acquisition gain | Return gain | Balanced score | Decision |
|---:|---:|---:|---:|---|
| 7210 | +0.463 points | 0.000 points | 0.000 points | stop |
| 7211 | **+1.157 points** | +2.778 points | **+1.157 points** | continue |
| 7212 | -1.620 points | -4.167 points | -4.167 points | stop |
| 7213 | +0.926 points | **+4.861 points** | +0.926 points | stop |

Clone 7211 cleared both positive-gain thresholds and had no harmful return
round.

Full completion preserved the ordering:

| Clone | Final reliability reward gain | Final return reward gain | Final conservative gain |
|---:|---:|---:|---:|
| 7210 | +0.463 points | 0.000 points | 0.000 points |
| 7211 | **+1.157 points** | **+6.250 points** | **+1.157 points** |
| 7212 | -1.620 points | -4.167 points | -4.167 points |
| 7213 | +0.926 points | +3.704 points | +0.926 points |

Selected 7211 reached 12.50% reliability accuracy (+6.94 target points) and
76.39% return accuracy (+63.89 points). Clone 7213 had a larger early return
signal but remained weaker on the conservative acquisition floor. Clone 7212
failed the full gate; all other clones retained both inherited primitives and
passed exactness/persistence gates.

## Adversarial control

Shuffling physical verifier-reward alignment on the winning seed made the full
gate fail. Reliability reward gain changed from +1.157 points to -3.472 points,
and the learned reliability target rate fell far below frozen. The intact
balanced result therefore depends on correctly aligned verifier outcomes.

## Accounting

The production race again used 180 rather than 216 physical rounds, saving
16.7%. No shadow lifetimes or shadow verifier bits were consumed. Completing
the three stopped clones and the reward-shuffled arm was post-gate validation
cost, not production allocation.

## Verdict

Promoted. Across two unseen prospective streams, the unchanged maximin selector
chose the clone with the greatest eventual conservative verified gain across
new-context acquisition and return retention. Both runs preserved inherited
skills and exact resumability; the replication additionally failed under
reward-alignment corruption.

This supports using balanced verifier gain as the population objective. The
next frontier is not a larger population yet. First test whether retaining the
selected weights improves the sample-efficiency curve on a genuinely later
held-out task compared with:

- the shared parent checkpoint;
- a fresh matched controller;
- and the architecture with selected weights reset.

That transfer ledger will determine whether population selection produces
compounding learning rather than merely selecting a good trajectory on the
current curriculum.

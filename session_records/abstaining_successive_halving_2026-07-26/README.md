# Prospective abstaining successive-halving race

Date: 2026-07-26

## Pre-registered protocol

Use unseen physical stream 7081 and clones 7170–7173. Preserve every learner
hyperparameter and both existing lexicographic rankings.

1. Train all four clones through round 18 and run the four-seed read-only
   acquisition audit.
2. Advance the top three. The fourth-ranked clone is the strongest-pruned
   validation control and independently resumes to round 54.
3. Resume the three survivors through round 42.
4. Rank survivors by mean learned-minus-frozen verifier reward over exactly the
   first six `old_return` rounds, then worst-round advantage, then lowest clone
   ID.
5. Resume the winner through round 54 only if its mean advantage is greater
   than `1e-6` and its worst-round advantage is at least `-1e-6`. These
   tolerances operationalize “strictly positive and no harmful round” without
   treating floating-point cancellation as evidence.
6. Otherwise abstain and spend no production compute on rounds 43–54.

No accuracy, target label, semantic ID, or hidden diagnostic enters selection.

## Accounting

- selected production path: 156 rather than 216 physical rounds (27.8% saved);
- abstaining production path: 144 rounds (33.3% saved);
- the one-time completed validation control adds 36 rounds to either figure;
- read-only shadow verifier bits and held-out lifetimes remain separate from
  learner training.

## Gate

If a winner is continued, it must beat the eliminated control in the direction
of reliability acquisition and old-return performance while retaining both
inherited primitives and passing every full gate.

If the race abstains, the decision is correct only if the completed eliminated
control also lacks a valid acquisition-and-return gain. A useful eliminated
control means the round-18 prune still misses sleepers and rejects the ladder.

## Results

### Acquisition screen

| Rank | Clone | Conservative shadow advantage | Decision |
|---:|---:|---:|---|
| 1 | 7170 | **+7.292 points** | advance |
| 2 | 7173 | +6.250 points | advance |
| 3 | 7171 | +3.125 points | advance |
| 4 | 7172 | 0.000 points | completed control |

No tie-break was used.

### Return screen and abstention decision

| Clone | Six-round mean reward advantage | Worst round | Decision |
|---:|---:|---:|---|
| 7170 | **+2.778 points** | 0.000 | continue |
| 7171 | +2.083 points | 0.000 | stop |
| 7173 | 0.000 points | 0.000 | stop |

Clone 7170 cleared both operational thresholds, so the race correctly did not
abstain.

### Completed comparison

| Measurement | Selected 7170 | Eliminated control 7172 |
|---|---:|---:|
| Reliability target accuracy | **59.72%** | 0.00% |
| Reliability target advantage | **+45.83 points** | 0.00 points |
| Old-return target accuracy | **51.39%** | 0.00% |
| Old-return target advantage | **+37.50 points** | 0.00 points |
| Old-return reward advantage | **+3.472 points** | 0.00 points |
| Binary retention | pass | pass |
| Four-rule retention | pass | pass |
| Full gate | pass | pass |

Every resumed prefix preserved all earlier trace rows exactly.

## Adversarial and selector audits

Shuffling verifier-reward alignment made the full gate fail and collapsed
old-return target accuracy to zero. The result therefore depends causally on
correct verifier outcomes.

A one-time shuffle of latent strategy keys did not collapse performance:
reliability rose to 73.61%, while old-return accuracy fell from 51.39% to
36.11%. This is evidence that key/value addressing affects retention, but not
that intact addresses are the sole source of the learned behavior. The agent
has 36 later rounds in which it can adapt after this intervention, so the
control must not be described as an immediate memory ablation.

After the gate was fixed, the two stopped survivors were completed as
diagnostics:

- 7171 reached 13.89% reliability accuracy but gained nothing on return;
- 7173 reached 44.44% reliability accuracy but gained nothing on return.

Thus the six-round selector chose the only survivor with a substantial
acquisition-and-return trajectory. These diagnostics cannot retroactively
change the production selection.

## Accounting

The production race used 156 physical rounds rather than 216, saving 27.8%.
The completed eliminated control added 36 validation rounds; completing the two
losers and the two adversarial arms was explicitly post-gate diagnostic cost,
not production cost. Each round-18 shadow audit used 112 held-out logical
lifetimes and 960 separately accounted selection verifier bits with no
optimizer updates.

## Verdict

Promoted provisionally. On an unseen stream, the conservative abstaining race:

- preserved a strong delayed learner;
- spent final-round compute only after a positive, non-harmful return signal;
- selected the best completed survivor;
- beat the strongest-pruned control in both acquisition and return;
- retained inherited skills;
- and failed under verifier-reward corruption.

The key-shuffle result narrows the memory-address claim but does not weaken the
population-selection result. Replicate the abstention rule once on another
unseen stream. A scientifically valuable replication may either select a
valid winner or correctly abstain when the completed control is also invalid.

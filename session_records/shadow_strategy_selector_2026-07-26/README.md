# Read-only shadow strategy selector

Date: 2026-07-26

## Motivation

Shortening or interleaving the successful 18-round context blocks damaged
learning. Existing round-18 signals were also insufficient on the blind 7074
population: cumulative early reward advantage selected clone 7090, while
action/information diversity favored harmful clone 7093. Clone 7092 eventually
had the strongest safe old-return gain.

## Audit

After the first unchanged 18-round block, every stored latent strategy is
evaluated read-only on held-out old-equal and reliability-dominant physical
contexts. The audit does not update controller weights, strategy memory, disk
memory, or context retrieval. It exposes no task labels or correct actions to
the learner. Only physical verifier rewards are used for population selection.

Four held-out audit seeds cost, per clone:

- 112 held-out logical lifetimes;
- 960 selection verifier bits;
- zero optimizer updates;
- exact physical/tensor parity.

The pre-registered primary score is the smaller of the two context-mean
best-slot reward advantages. Context-specific best-slot count is inspected only
as a tie-break diagnostic.

## Retrospective blind result

| Clone | Conservative shadow advantage | Seeds with different best slots | Later return reward advantage | Later return target advantage | Final gate |
|---|---:|---:|---:|---:|---:|
| 7090 | +2.08 points | 0/4 | 0.00 points | 0.00 points | pass |
| 7091 | 0.00 points | 0/4 | +0.46 points | +2.78 points | pass |
| 7092 | +2.08 points | 2/4 | +3.01 points | +18.06 points | pass |
| 7093 | 0.00 points | 0/4 | -0.46 points | -8.33 points | fail |

The primary score safely prunes 7091 and harmful 7093 but ties 7090 and 7092.
The context-specialization tie-break selects 7092, the eventual best safe
return clone.

## Prospective result

The rule was frozen before opening physical stream 7075. Four new clones
(7100–7103) were screened after the unchanged first 18-round block:

| Clone | Conservative shadow advantage | Specialized audit seeds |
|---|---:|---:|
| 7100 | +2.08 points | 0/4 |
| 7101 | **+10.42 points** | 0/4 |
| 7102 | 0.00 points | 0/4 |
| 7103 | +1.04 points | 1/4 |

The primary rule selected 7101 without a tie-break. Only 7101 and the
pre-fixed lowest-ID control 7100 were extended to 54 rounds.

| Arm | Reliability target | Old-return target | Return reward advantage | Final gate |
|---|---:|---:|---:|---:|
| selected 7101 | **73.6%** | **81.9%** | **+5.79 points** | pass |
| fixed control 7100 | 8.3% | 2.8% | +0.23 points | pass |
| 7101, rewards shuffled | 20.8% | 36.1% | +0.23 points | fail |
| 7101, strategy keys shuffled | 30.6% | 0.0% | +0.69 points | pass |

All arms retained binary and four-rule capability. Reward shuffling destroyed
the overall result; shuffling strategy keys at transfer erased old-return
behavior. The selected gain therefore depends causally on aligned verifier
outcomes and correct latent-memory addressing.

## Verdict

Promoted as the current population selector. It preserves the learning
trajectory, spends extra verifier compute rather than extra optimizer updates,
and prospectively selected a vastly more transferable clone. The current
evidence is one prospective four-clone population; magnitude must replicate on
another physical stream before increasing population size or audit budget.

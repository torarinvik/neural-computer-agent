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

## Verdict

Promising candidate selector, not yet promoted. This is retrospective on one
four-clone blind population. The next gate is prospective: pre-register the
same lexicographic rule on a fresh physical stream, extend only its selected
clone plus a blinded control, and require improved return without retention
loss. A reward/context-shuffled audit is required if the prospective result is
positive.

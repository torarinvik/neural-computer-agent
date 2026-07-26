# Prospective balanced maximin population race

Date: 2026-07-26

## Pre-registered protocol

Use unseen physical stream 7084 and clones 7200–7203. Train all clones through
round 42 with no shadow audit and no early elimination.

For each clone compute only from verifier rewards already produced by its
physical trajectory:

```text
acquisition_gain =
    reliability-phase mean learned reward - frozen reward

return_gain =
    first-six-return mean learned reward - frozen reward

balanced_score = min(acquisition_gain, return_gain)
```

Rank by descending balanced score, then descending return gain, then descending
acquisition gain, then ascending clone ID.

Continue the winner only if:

- acquisition gain is greater than `1e-6`;
- return gain is greater than `1e-6`;
- every one of the six return-round reward advantages is at least `-1e-6`.

Otherwise abstain. A continued winner receives exactly twelve more rounds.

## Validation

For this first prospective maximin test, complete all stopped clones after the
selection decision. Diagnostics cannot alter the winner and are excluded from
production accounting.

Promotion requires the selected clone to have the greatest final conservative
verified gain:

```text
min(final reliability reward advantage, final return reward advantage)
```

among valid completed clones, while preserving inherited skills and every
exactness/persistence gate. An abstention is valid only if no completed clone
has positive reward gain in both phases.

## Results

At round 42:

| Clone | Acquisition gain | Return gain | Balanced score | Decision |
|---:|---:|---:|---:|---|
| 7200 | +1.389 points | +2.083 points | +1.389 points | stop |
| 7201 | 0.000 points | 0.000 points | 0.000 points | stop |
| 7202 | +0.231 points | 0.000 points | 0.000 points | stop |
| 7203 | **+1.620 points** | **+3.472 points** | **+1.620 points** | continue |

Every return round for 7203 was non-harmful, so it cleared the continuation
gate.

The completed final comparison was:

| Clone | Final reliability reward gain | Final return reward gain | Final conservative gain |
|---:|---:|---:|---:|
| 7200 | +1.389 points | +0.926 points | +0.926 points |
| 7201 | 0.000 points | 0.000 points | 0.000 points |
| 7202 | +0.231 points | 0.000 points | 0.000 points |
| 7203 | **+1.620 points** | **+2.083 points** | **+1.620 points** |

Selected 7203 remained the best balanced completed learner. It reached 22.22%
reliability accuracy (+19.44 target points) and 15.28% old-return accuracy
(+13.89 points). Clone 7200 was a valid but weaker balanced learner. Every
clone passed inherited retention, persistence, physical/tensor parity, and
exact continuation gates.

## Accounting

The production path used 180 physical rounds instead of 216, saving 16.7%.
There were no shadow held-out lifetimes or shadow verifier bits. Completing the
three stopped clones was a one-time post-gate validation cost and is excluded
from production accounting.

## Verdict

Provisionally promoted. The verifier-only maximin score selected the clone with
the best eventual conservative gain across both acquisition and return. This
is the first allocation rule in the sequence whose optimization target
directly matches compounding learning rather than one phase alone.

Magnitude is moderate and this is one prospective stream. Replicate unchanged
before increasing population size, task difficulty, or training duration.

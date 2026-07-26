# Conservative three-of-four successive halving

Date: 2026-07-26

## Pre-registered protocol

Test the evidence-directed repair on unseen physical stream 7080 with clones
7160–7163. All learner hyperparameters, the round-18 acquisition ranking, and
the round-42 six-return-round ranking remain frozen.

The only change is pruning pressure:

1. advance the top three acquisition clones at round 18;
2. designate the sole eliminated clone as the validation control and resume it
   independently to round 54;
3. resume all three survivors to round 42;
4. advance the one six-round retention winner to round 54.

Production cost is 156 versus 216 physical rounds, a 27.8% saving. Including
the one-time eliminated-control audit costs 192 rounds, still 11.1% below
exhaustive completion.

## Gate

Promote only if the selected winner exceeds the eliminated control in the
direction of both reliability acquisition and old-return performance, retains
both inherited primitives, and passes all exactness, parity, persistence, and
graduation gates.

A tie or loss rejects even this conservative prune. Do not increase population
size or training duration on a rejection.

## Results

The acquisition ranking was:

| Rank | Clone | Shadow advantage | Specializing seeds | Decision |
|---:|---:|---:|---:|---|
| 1 | 7163 | +6.250 points | 1/4 | advance |
| 2 | 7161 | +2.083 points | 1/4 | advance |
| 3 | 7160 | +2.083 points | 0/4 | advance |
| 4 | 7162 | 0.000 points | 0/4 | validation control |

The six-round retention score was:

- 7160: mean 0, worst 0;
- 7161: mean 0, worst 0;
- 7163: approximately 0 mean due one +4.17 and one -4.17 round, worst
  -4.17.

The frozen lowest-ID tie-break selected 7160. It completed with zero
reliability and zero old-return advantage. Eliminated control 7162 also
completed with zero in both phases. The pre-registered gate therefore
rejected the run.

## Post-gate localization

The two non-selected survivors were completed only after the gate result, as a
diagnostic that could not alter the declared winner:

- 7161 gained +1.39 reliability target points but zero old-return advantage;
- 7163 gained +2.78 reliability and +1.39 old-return target points, but its
  reliability reward was 0.463 points *below* frozen and its full gate failed.

No member of this population produced a valid acquisition-and-return winner.
The zero selected-control result is therefore not evidence that the
three-of-four prune discarded a good clone. It is evidence that the selector
should be allowed to abstain when all six-round return scores are non-positive.

## Verdict

Do not scale. Three-of-four pruning repaired the known stream-7079 sleeper
failure retrospectively, but this fresh population contained no candidate that
passed the full criterion.

The next minimal experiment should add an abstention gate without changing the
learner:

- advance three of four at round 18;
- at round 42, continue a winner only when mean six-round verified reward
  advantage is strictly positive and worst-round advantage is non-negative;
- otherwise stop the whole population and spend no final-round compute.

This rule would have continued the useful winners on streams 7077–7079 and
abstained on stream 7080. It must be pre-registered and tested prospectively
before promotion.

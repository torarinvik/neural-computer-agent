# Reliability-context ramp

Date: 2026-07-26

## Question

Does replacing repeated contexts with a gradual reliability ramp produce more
action-informative soft-retrieval comparisons at the same verifier budget?

## Design

- six physical rounds;
- reliability weights: 0.0, 0.1, 0.2, 0.3, 0.4, then 0.0;
- recency and frequency remained equal and shared the remaining weight;
- 13 context parameters, four strategy slots;
- SPSA perturbation 1.2 and softmax temperature 0.08;
- seed 7072, four banks, capacity six;
- no semantic labels or hidden state supplied to the learner.

The report now records action-divergent pairs, reward-divergent pairs, their
fractions, mean reward delta, unique utility contexts, and verifier bits per
informative pair.

## Result

- five unique utility contexts;
- one of five eligible pairs changed actions (20%);
- the same one of five changed physical reward (20%);
- reward difference: 4.17 percentage points;
- 672 verifier bits per reward-divergent pair;
- learned context scales moved by about 2.1%;
- binary and four-rule retention passed.

This exactly matches the informative-pair density of the earlier best sharp
screen. More reliability contexts did not make verifier credit denser.

## Verdict

Rejected as a scaling direction. Do not lengthen or replicate this curriculum.

Two follow-ups also failed to improve the rate:

- temperature 0.04 instead of 0.08: one informative pair of five;
- 16 cost-free perturbation proposals per decision: one informative pair of
  five after screening 80 proposals.

A bank-level action probe localized the failure further. The stored strategies
briefly produced two distinct action patterns at the only informative round,
then collapsed to one behaviorally distinct pattern for the remaining
contexts. The next fork must improve *behavioral strategy diversity*, not add
contexts, perturbation directions, encoder parameters, or training time.

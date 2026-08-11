# Recency-plus-latest transition routing diagnostic

Date: 2026-08-11
Status: **rejected diagnostic; no promotion**
Schema: `neural-computer.brainworkshop-factored-residual-sequence-pressure.v1`

## Question

Does a context key that preserves both recency-weighted evidence and the
actual latest reliable event improve the three-regime, ten-step external
factored-memory pressure test?

The controller and affine base remained frozen. Each target residual was
selected by copy-on-write held-out promotion. No optimizer updates or replay
were used.

## Matched result

The new `recency_weighted_and_latest` aggregation was compared with the
compatibility `last_token` aggregation on seeds 91, 92, and 93, with identical
budgets and thresholds.

| condition | complete | regime promotions | full runs | missing-evidence safety | replay | optimizer updates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `last_token` | 0/3 | 8/9 | 2/3 | 0/3 | 0 | 0 |
| `recency_weighted_and_latest` | 0/3 | 8/9 | 2/3 | 0/3 | 0 | 0 |

The serialized reports were byte-identical in their outcome fields. Seed 92
could not stage the third regime; seeds 91 and 93 staged and promoted all
three. The cumulative partial-evidence route remained unresolved on every
seed. The controller stayed unchanged, the base stayed frozen, and checksum
corruption was rejected on both fully promoted runs.

## Interpretation

This is not evidence that recency is harmful, but it is not a capability gain.
The bottleneck is currently downstream of simple context aggregation:
long-horizon factual error accumulates as evidence is extended, and the
read-only route refuses rather than guessing. That refusal is the correct
safety behavior. Do not weaken the ambiguity gate to manufacture a promotion.

The next experiment should isolate the two causes: evaluate one-step factual
identity separately from recursive ten-step model error, then test a
horizon-aware evidence verifier or a bound-once transition context without
relaxing contradiction refusal.

## Accounting

- Unique verifier bits: 338 total (122 + 94 + 122).
- Logical lifetimes: 54 total (19 + 16 + 19).
- Transition rows consumed once: 270.
- Optimizer updates: 0.
- Replayed examples: 0.
- Full-gate passes: 0/3.
- Claim boundary: this is a routing/addressing diagnostic, not a claim of
  general continual learning.

Reports and the checksum manifest are stored beside this README.

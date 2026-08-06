# Diagnostic: BPTT truncation window is not the acquisition bottleneck (2026-08-06)

Identical Snake configurations trained at detach intervals 1 and 8 with
matched budgets and seeds.

| metric | seed 69316 (k=1 / k=8) | seed 69317 (k=1 / k=8) |
| --- | ---: | ---: |
| endpoint mastery | 0.8555 / 0.8457 | 0.9297 / 0.9414 |
| mean training mastery | 0.6361 / 0.6941 | 0.6640 / 0.6875 |
| updates to half mastery | 93 / 93 | 131 / 101 |

Eight-step credit through the recurrent core changes nothing meaningful
for reactive games: the one-step regime used by every promoted rung is
vindicated, and the shared-controller acquisition gap (0.86-0.93 vs 0.94
standalone) must be attributed to the recurrent optimization landscape or
capacity, not gradient truncation. Zero replay in all conditions.

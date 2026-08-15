# Current-symbol prototype-match acquire (2026-08-15)

Status: **replicated, not admitted**.

Unused seeds 116017, 117017, and 118017 each acquired a prototype-match
file on rendered current-symbol and held it frozen. `AgentBrain.bank` was
not written. Slot 0 stayed `90e20193…`. This is not a Dual holdout and
does not consume that lease.

## Results

| Seed | Train | Frozen hold | Zeros | Reverse | Shuffle | Delay slot 0 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 116017 | 0.938 / 48 | 1.000 / 48 | 0.500 | 0.000 | 0.396 | 0.708 |
| 117017 | 0.958 / 48 | 1.000 / 48 | 0.500 | 0.000 | 0.521 | 0.750 |
| 118017 | 0.938 / 48 | 1.000 / 48 | 0.500 | 0.000 | 0.583 | 0.688 |

Controller digest stayed `59c9ef2b…`. Program-file updates were 48 per
seed (144 total). Optimizer updates and replay were zero.

## Limits

- not a curated bank slot;
- not open program induction;
- not a Dual 2-back transfer ratio;
- not desktop Dual.

A later control binds prototype files to a frontend digest. The same
template fails on a different random encoder. Delay-address files stay
frontend-agnostic. That is why this campaign did not write AgentBrain.

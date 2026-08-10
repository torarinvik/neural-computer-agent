# Scaling to N=1024, the project's own reacher, and a fixed null (F92-F94)

## F92 — the reacher, read by a plant that never saw a grid

| family | read (0 gradient steps) | fine-tune | cold |
| --- | ---: | ---: | ---: |
| grid (open, r3) | 1.000 | 0 | 50 |
| walled (r4) | 0.894 | 438 | 88 |

The open grid is acquired FREE — first transfer in this project to a task it
was built for rather than one built for it. The walled grid is the first
decisive FAILURE: fine-tuning costs 5x cold. Cause, predicted before the run:
every generated op is a uniform function of SLOT VALUES, while a wall makes an
action's effect depend on WHICH STATE you are in. Walls change 27/256
transitions and those are exactly the ones the plant cannot represent. Neither
--wide nor --balanced reaches this; the missing primitive is conditional
effects.

## F93 — the wrong-context null was broken, and is fixed

Pairing each family with its LIST NEIGHBOUR stopped being a null once
near-duplicates entered the set: grid's neighbour is walled, which shares
229/256 transitions, so grid's null read 1.000.

| family | read | stranger entry | neighbour (old) | withheld |
| --- | ---: | ---: | ---: | ---: |
| grid | 1.000 | 0.117 | 1.000 | 0.170 |
| walled | 0.894 | 0.156 | 0.350 | 0.221 |
| toggle | 0.917 | 0.021 | 0.227 | 0.000 |

Controls must be drawn at random from the generating distribution, not from the
neighbourhood.

Clause (c) at N=1024: 1024/1024 and 1023/1024 mastered, retention drift 0.0,
acquisition first-128 4.3/8.8 vs last-128 5.3/10.5 against cold ~49.

## F94 — retrieval to N=1024

| N | key | key+verify | scan | key gap | conseq gap | stranger key sim |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 1.000 | 1.000 | 0.969 | 0.128 | 0.356 | 0.862 |
| 256 | 0.988 | 0.994 | 0.918 | 0.068 | 0.259 | 0.923 |
| 1024 | 0.951 | 0.980 | 0.853 | 0.037 | 0.171 | 0.954 |

Retrieve-then-verify: 0.980 at N=1024 on a CONSTANT 4 plant passes vs the
1024-pass scan at 0.853.

**Correction to F87.** F87 claimed the key-gap decrements were decelerating
toward an asymptote. Decrements are -0.109, -0.049, -0.038, -0.033, -0.027,
-0.002, -0.029 — still falling at the earlier rate. Having criticised a linear
extrapolation from four points, F87 then made an asymptotic claim from the same
four. Both were projections dressed as findings.

Key-only discrimination is effectively gone at N=1024 (stranger similarity
0.954). Ranking survives (0.951) and consequence verification survives (gap
0.171), so the verify step's importance GROWS with bank size.

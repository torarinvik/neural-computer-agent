# The world changes, the agent notices, and remembers the old one (2026-08-15)

Status: **diagnostic**, adversarially calibrated. Nothing admitted;
`AgentBrain.bank` unchanged at `07319eb1`.

Non-stationarity was the last adversarial probe still standing. Every learner
in this session assumes a fixed target, and a rule that changes partway defeats
all of them. Outside a benchmark it is the normal case rather than the
exception, and an agent that cannot survive it is not an agent.

Two capabilities are needed, and they are not the same one.

**Detection** -- knowing the current hypothesis has stopped working, and
telling that apart from the noise it was already tolerating. This became
possible only after noise tolerance, because it needs a *calibrated*
disagreement rate rather than a demand for perfection. A regime change is a
recent disagreement rate too high to be the noise already accounted for: a
binomial tail test on evidence already paid for.

**Recognition** -- not relearning a regime seen before. This is where the
accumulation work pays off outside its own record. The library of fitted
machines is consulted before any new fit, so a world that oscillates costs a
fit per *distinct* regime rather than per switch.

## Result

Twenty-seven stationary streams and twenty-four changing ones, at three noise
levels, over nine sampled rules of two to four states.

| | 0% noise | 5% noise | 10% noise |
| --- | ---: | ---: | ---: |
| False change-points per 48-episode stationary stream | 0.11 | **0.04** | **0.04** |
| Change detected | 8/8 | 8/8 | 8/8 |
| Detected **exactly** at the true episode | 8/8 | 8/8 | 8/8 |
| Spurious extra splits on a changing stream | 2 | **0** | **0** |
| Returning regime recognised | 8/8 | 7/8 | 7/8 |
| Fits spent on three regimes (two distinct) | **2.12** | **2.12** | **2.12** |

Every change is found, and found at the episode it happened. A world that goes
A, B, A costs **2.12 fits for three regimes** -- the return is recognised, not
relearned.

## Two constants that were wrong, and the measurements that said so

Both thresholds started as fixed numbers and both had to become *relative to
the noise*.

**Reuse allowance.** A fixed 0.05 can never accept a correct machine on a
stream carrying 10% label noise, because a correct machine disagrees with 10%
of the labels. Recognition fell to **0/8** on exactly the streams that needed
it. Made relative to the running noise estimate, it is 7/8.

**The reference rate for detection.** Comparing a window against the error on
the four episodes the machine was *fitted* to underestimates the noise, so
later windows look anomalous. That produced up to **0.81** spurious change
points per stationary stream. Comparing against the running error over the
whole current segment took it to 0.26.

## The threshold, swept rather than chosen

At 1e-3, 1e-5 and 1e-8, over 216 stationary streams and 72 changing ones:

| alpha | false alarms/stream | detected | exact | reuse |
| --- | ---: | ---: | ---: | ---: |
| 1e-3 | 0.25 | 24/24 | 24/24 | 22/24 |
| 1e-5 | 0.09 | 24/24 | 24/24 | 22/24 |
| **1e-8** | **0.06** | 24/24 | 24/24 | 22/24 |

Tightening is free in this regime, which is worth understanding rather than
just taking: a real regime change is not a marginal deviation. The machine that
fitted the old world is simply wrong about the new one, so the tail probability
collapses and no plausible threshold misses it. The default is 1e-8.

## What is honestly weak

**False alarms are not zero.** About one spurious split every nine to
twenty-five stationary streams. Each costs a wasted fit and splits one regime
into two, which is recoverable -- the second half is recognised by the library
-- but it is not free.

**Detection is at window granularity.** Change points land on multiples of the
window, four episodes here. A change is never missed but it is never located
finer than that.

**Recognition is behavioural, not semantic.** A library machine is accepted
because it predicts the window, not because anything establishes it is the same
rule. On these regimes that coincides; on a distribution where two rules agree
locally and diverge later it would not, and nothing here would notice.

**The switching probe itself is still not learned.** It alternates *within*
every episode, so there is no stretch of episodes for a regime to be fitted on.
This handles change *between* episodes. Sub-episode non-stationarity remains
open.

**Nothing is admitted, and the controller executes none of this.**

## Reproducing

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_regime_tracking.py -q
```

Seconds. The calibration sweeps are `calibration.txt` in this record.

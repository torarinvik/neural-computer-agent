# The agent notices its hypothesis class is too small, and leaves it (2026-08-15)

Status: **diagnostic**, adversarially calibrated. Nothing is admitted and
`AgentBrain.bank` is unchanged at `07319eb1`.

The adversarial audit left this flaw open and named it the worse of the two.
Given a running-majority rule -- press while one symbol has occurred more often
than another -- the inducer returned a **twelve-state machine scoring 0.518**
on held-out evidence. It did not abstain and it did not warn. Majority is not
finite-state at any size, so no amount of search or data inside that class
would ever have helped, and nothing in the stack could tell.

That is the corner-painting failure in its purest form. Every other advance in
this session -- proposers, dedup, feedback ranking, factorisation, episode
segmentation -- is search *within* a fixed hypothesis class. None of it
touches this.

## Two pieces

**A competence verdict, read off the learning curve.** Fit at each rung of an
evidence ladder and watch two trajectories: the size of the hypothesis and its
error on episodes it was not fitted on. Inside the class they settle. Outside
it, every rung buys another state and the error stalls. The verdict uses no
knowledge of the answer, which is what makes it usable by an agent rather than
by an experimenter.

**An escalation gated on held-out evidence.** The class widens by exactly one
construct: a single integer counter moved by -1, 0 or +1 per transition, whose
*sign* the output may read. Deliberately the smallest step out of finite state
rather than a leap to a universal machine -- a class that can express anything
can fit anything, which would destroy the verdict it is supposed to serve.
Increments are tried `0` first, so the wider class does not reach past finite
state without cause.

## The result

Running majority, at 48-step episodes:

| | outcome |
| --- | --- |
| Finite-state inducer | 12 states, **0.518** held-out |
| Escalated counter machine | **1 state, 0.0000** held-out |

Exact, in under a tenth of a second, from 112 short episodes -- and verified on
a third set of episodes neither the fit nor the gate had seen.

## Calibration

Thirty-two targets -- twenty-four sampled Mealy rules of one to six states,
four running-majority rules, three thresholds and parity -- at three episode
lengths.

| Episode length | in-class identified | out-of-class escalated | **false IDENTIFIED** |
| ---: | ---: | ---: | ---: |
| 16 | 28/28 | 3/4 | **0** |
| 48 | 28/28 | **4/4** | **0** |
| 128 | 28/28 | **4/4** | **0** |

**Ninety-six verdicts, zero of them a false claim of success.** Every
`IDENTIFIED` came with exactly zero held-out error. No in-class target ever
escalated, at any length.

The one out-of-class target not escalated at length 16 was correctly
`IDENTIFIED` instead: over sixteen steps a bounded counter *is* finite-state,
so the label was wrong rather than the verdict.

## Adversarial probes it survived

**Structureless data.** Random labels over random symbols, at 7, 28 and 112
episodes: the counter class finds nothing every time. A class that fits noise
diagnoses nothing.

**Real Mealy rules of three to six states.** The counter class correctly finds
nothing, so escalation cannot quietly take over targets the narrower class
owns.

**Label noise at 1% and 5%.** On both an in-class and an out-of-class target,
every verdict degrades to `NEED_MORE_DATA` and nothing is accepted. The stack
still cannot learn under noise -- that flaw is untouched -- but it does not
lie about it.

## What is honestly weak

**`CLASS_INADEQUATE` is nearly useless on its own.** It fires on one of four
majority rules at sixteen-step episodes and on none at forty-eight or a
hundred and twenty-eight, because the finite-state search stops returning
anything and a missing trajectory cannot be read. From the inside, abstention
and inadequacy are indistinguishable.

So the trigger that actually does the work is the weaker one: *the finite-state
class did not explain the evidence, whatever the reason.* What makes that safe
is not the trigger but the gate -- a wider hypothesis is accepted only when it
predicts episodes it was never fitted on. The calibration table is the evidence
that the gate holds, not an argument that the trigger is clever.

**The classes are complements, not nested.** The counter class searches at most
two control states, so it identifies parity and majority and fails on
threshold-3 and threshold-6, which the finite-state class gets. Escalation adds
reach; it does not subsume.

**Non-stationary targets defeat both.** A rule that switches halfway through
each episode is identified by neither and correctly claimed by neither.

**Noise remains unsolved.** Everything above is at zero label noise.

## Reproducing

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_class_escalation.py -q
```

About five minutes; the calibration sweep is `calibration.txt` in this record.

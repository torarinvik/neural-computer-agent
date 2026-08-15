# Onset lease at the trial floor (2026-08-15)

Status: **replicated, not admitted**. Pre-registered in
`session_records/PREREGISTRATION_discriminating_leases_2026-08-15.md`, which
was committed before this ran and predicted `and` on every seed.

Unused block `onset_lease_discriminating` (134017, 135017, 136017) at 448
steps, so 447 eligible trials per session. `AgentBrain.bank` was not written
and nothing was admitted.

## Results

Winner is `and:0` on every seed, bound to frontend `1ce405a0…`. Controller
digest `59c9ef2b…`. Slot 0 stayed `90e20193…`.

| Seed | Acquire | Six frozen holds | Stable prefix | retrieve slot 0 | invert slot 0 | prototype only | zeros | reversed | reward shuffled | other encoder |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 134017 | 0.749 | 1.000 × 6 (2682 bits) | 447 | 0.251 | 0.749 | 0.729 | 0.749 | 0.000 | 0.497 | 0.749 |
| 135017 | 0.749 | 1.000 × 6 (2682 bits) | 447 | 0.251 | 0.749 | 0.779 | 0.749 | 0.000 | 0.501 | 0.749 |
| 136017 | 0.749 | 1.000 × 6 (2682 bits) | 447 | 0.251 | 0.749 | 0.745 | 0.749 | 0.000 | 0.492 | 0.749 |

Every pre-registered criterion held: predicted winner on every seed, bound
frontend, stable prefix, zero program-file updates after the acquire session,
every reject control below `0.8`, controller digest unchanged, bank byte
identical.

## What carries the claim

Onset needs two families. Neither the admitted delay file (`0.251`) nor its
inverse (`0.749`) nor the acquired prototype alone (`0.729`-`0.779`) clears
`0.8`; only `and(invert(slot 0), prototype)` solves it, from a two-phase
acquire that labels change trials with the inverse and then averages the
invert-matching, reward-1 events into the template.

The winner did not merely clear the gate, it scored **447 of 447 on every
session of every seed**. A policy with a true rate of `0.78` produces one such
perfect session with probability `6e-49`. That, not the `0.8` gate, is what
makes this hard to explain as luck.

## Separation, and one claim that stays weak

Acceptance now requires the winner to be separated from its **strongest**
rejected arm, not merely for the two to land on opposite sides of `0.8`. Per
seed, the weakest held session against the best control:

| Seed | Best control | Its rate | Margin | P(that arm reproduces the winner's run) |
| ---: | --- | ---: | ---: | ---: |
| 134017 | zeros | 0.749 | 0.251 | 1.0e-56 |
| 135017 | prototype only | 0.779 | 0.221 | 2.5e-49 |
| 136017 | zeros | 0.749 | 0.251 | 1.0e-56 |

This gate is what the 48-step campaign actually failed: its spurious AND at
`0.812` against a `0.75` arm over 47 trials reproduces one time in five.

**The weak claim.** "No single family clears `0.8`" is a statement about each
control's *true* rate, and observing one under the gate does not establish it.
On 135017 the prototype-only arm was seen at `0.779`, which is what a true
`0.8` arm produces `14%` of the time, so that seed does not rule out a
single-family solution at the gate. The other five control-seed pairs here do
(`p <= 0.0054`). Every replicate records this as
`control_below_threshold`; it is reported, not gated, because acceptance rests
on necessity — the AND beats every rival by 0.22 or more with perfect holds —
rather than on where an arbitrary constant falls.

Closing that gap needs either a longer episode (ruling out a `0.78` arm at 1%
costs `2326` trials, roughly five times these) or a task variant where the
prototype-only base rate is further from the gate.

## Not claimed

- no admission: the AND child is not a curated `AgentBrain.bank` slot;
- no bits-to-threshold transfer ratio against a fresh learner;
- no learned proposer; search still enumerates the closed grammar;
- depth-2 files were proposed but are not installable on the
  prototype-capable machine, so coverage was retrieve {0,1}, invert {0,1},
  `and`, invent;
- no new operator types beyond retrieve, compose, invert, and, invent.

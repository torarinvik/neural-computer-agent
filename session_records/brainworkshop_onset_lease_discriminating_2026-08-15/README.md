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

## Where the floor is and is not enough

At 447 eligible trials a `0.75` policy reaches `0.8` with probability
`0.0066`, which clears the pre-registered 1% floor. That floor is stated
against a `0.75` near miss, and it does not cover everything:

- the prototype-only control was observed at `0.779` on 135017. If its true
  rate were `0.78`, it would cross `0.8` about **15.6%** of the time even at
  this length;
- pushing the floor out to a `0.78` near miss costs `2326` eligible trials,
  roughly five times these episodes;
- so this record's strength comes from the winner-control gap (`0.221` to
  `0.271`) and the perfect holds, not from the control margins alone.

Tightening the rule to gate on that gap, rather than on both sides crossing a
fixed constant, is the open protocol question.

## Not claimed

- no admission: the AND child is not a curated `AgentBrain.bank` slot;
- no bits-to-threshold transfer ratio against a fresh learner;
- no learned proposer; search still enumerates the closed grammar;
- depth-2 files were proposed but are not installable on the
  prototype-capable machine, so coverage was retrieve {0,1}, invert {0,1},
  `and`, invent;
- no new operator types beyond retrieve, compose, invert, and, invent.

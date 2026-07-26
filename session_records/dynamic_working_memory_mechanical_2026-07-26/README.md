# Dynamic working-memory mechanical gate

Date: 2026-07-26

## Scope

This is a plumbing result, not a capability claim.

An eight-slot physical RAM pool now exists alongside the unchanged four-slot
latent-strategy control. It supports a generic score-derived active mask,
explicit reads and writes, occupancy and eviction accounting, and exact
save/reload. Learned admission and eviction are intentionally not connected to
behavior yet.

The accompanying capability ledger records verifier bits, unique lifetimes,
updates, replay, candidate evaluations, memory traffic and occupancy, thought
steps, disk traffic, latency, and GPU time.

## Mechanical evidence

- Focused suite: 26 passed in 0.59 seconds.
- Full repository suite: 299 passed, 3 warnings, 15 subtests passed in 18.16
  seconds.
- A fixed four-slot active mask inside the eight-slot pool is available as the
  matched control.
- Save/reload preserves values, occupancy, active mask, usage, age, clock, and
  operation statistics exactly.

## Gate

Pass. The next experiment may train only the tiny context encoder. Dynamic
admission, eviction, and resource rewards remain disabled until intact context
keys beat shuffled keys on held-out verifier reward across at least two seeds
and an equal-probe control.

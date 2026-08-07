# Rejected: sequential weight-learning through shared drivers (2026-08-07)

The architecture-true driver configuration (one screen encoder, one 4-key
keypress decoder, one feedback encoder for all games) was tested on the
three-game arbitrated ladder. Findings across the probe series and one
full-budget run (seed 69317):

- The shared screen encoder HELPS first-game acquisition (Snake 0.9219
  at full budget, above the per-game 0.9355 within noise; 0.625 vs 0.484
  at small budget).
- Whole-plant consolidation (controller + shared drivers under one
  arbitrated rule) solves shared-driver retention: Snake 0.9043 after two
  later phases, versus 0.2266 with unprotected drivers.
- But the SECOND game cannot acquire through the occupied shared decoder
  at any tested configuration: Pong 0.1875 at full budget (per-game
  baseline: 0.9824/0.8203), 0.20-0.28 at small budget, worse at higher
  release (mu 6: 0.047). Neither budget nor the arbitration dial is the
  binding constraint - the interference is structural: a one-pass core
  cannot context-switch one decoder mapping across games it learned
  sequentially in weights.

## Implication (recorded steering decision)

Shared drivers and weight-resident sequential skills are incompatible.
The architecture's own answer is skill-as-context: the externalization
line already trains a core to behave differently per fetched artifact,
which is exactly the per-game decoder-mapping context switch this rung
lacks. The bank-stored ladder with shared drivers is therefore not an
independent steering item but the REQUIRED path: artifacts provide the
context that makes one decoder serve many games.

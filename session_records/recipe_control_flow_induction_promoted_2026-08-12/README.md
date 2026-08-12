# Promoted bounded from-scratch control-flow induction

This audit removes the hand-shaped executable scaffold used by the prior
control-flow rung. A generic finite enumerator explores every four-instruction
program over a two-counter basis with increment, decrement, jumps, conditional
jumps, and halt. The private verifier requires a loop that clears an opaque
counter and returns only scalar exact-match outcomes.

Across seeds `17`, `18`, `19`, and `20`, with both forward and reversed input
orders, the search found an executable equivalent to the target within `36`
candidate evaluations. Held-out and reload accuracy were `1.0000`; the
protected straight-line source file was retained; missing evidence, shuffled
feedback, and corrupted payloads were rejected; and the ten-candidate control
returned `budget_exhausted` rather than falsely claiming `inexpressible`.
Replay and optimizer updates were zero.

This promotes bounded from-scratch loop induction in a finite generic search
space. It does not prove efficient arbitrary program synthesis, unrestricted
execution, universal transfer, or general continual learning. The next
pressure is scaling program length and composing multiple learned control-flow
files without falling back to exhaustive enumeration.

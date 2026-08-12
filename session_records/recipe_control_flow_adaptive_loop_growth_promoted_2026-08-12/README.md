# Replay-free adaptive growth through non-commuting control-flow loops

This audit extends the adaptive external CPU/files curriculum beyond
straight-line increments. The root is an opaque counter-clearing loop. Three
successive verifier-qualified files add one increment to the loop body at a
time, growing the program from length four through lengths five, six, and
seven. Inserting an instruction must relocate existing jump targets; a stale
numeric target produces a syntactically valid but wrong program.

Across seeds `17–20`, both forward and reversed verifier-state orders reached
all three loop rungs. Every new loop scored `1.0000` on held-out initial
states, and every earlier loop remained at `1.0000` after later growth. State
and external program-memory reloads were exact; missing evidence and corrupted
state were rejected without mutation; shuffled feedback qualified zero rungs;
and replayed examples and optimizer/controller updates were zero.

The matched fresh source-to-loop control reached the first rung in every seed,
but failed to discover the final length-seven target within 1,200 candidate
evaluations in every seed. This is evidence for curriculum value, not a formal
warm/fresh transfer ratio because the fresh final gate did not cross.

The positive arms charged `20,270` verifier bits across `4,054` candidate
lifetimes; fresh controls charged `45,695` bits across `9,139` lifetimes; and
shuffled controls charged `9,600` bits across `1,920` lifetimes. The complete
audit charged `75,565` bits and `15,113` logical lifetimes, with zero replay
and zero optimizer updates.

This promotes bounded replay-free adaptive growth of non-commuting external
loop programs with retention. It does not establish efficient arbitrary
program synthesis, unrestricted execution, unrestricted memory growth, or
general continual learning.

The runnable audit is
`experiments/recipe_expressibility/control_flow_adaptive_loop_growth.py`.

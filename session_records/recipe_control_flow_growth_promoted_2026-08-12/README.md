# Promoted generic external control-flow growth

This audit adds the first non-straight-line computation boundary to the
external recipe substrate. `ControlFlowProgram` is a generic two-counter-style
machine with increment, decrement, unconditional jump, zero/nonzero
conditional jump, and halt. Executions are bounded by explicit resource
limits; program files are stored outside the controller in a checksummed,
protected external memory.

The verifier privately requires a loop that transfers an opaque counter value
to one destination. The learner starts from a protected transfer scaffold,
receives only scalar exact-match outcomes, and uses
`ControlFlowOutcomeSearch` to propose one generic instruction edit. The
acquired program changes the destination counter, reaches exact held-out
mastery, and is stored as a second protected file without changing the source.

Across seeds `17`, `18`, `19`, and `20`, both forward and reversed input orders
promoted. Held-out accuracy and reload accuracy were `1.0000` in every arm;
source retention was `1.0000`; shuffled feedback never admitted a file;
missing evidence left the memory digest unchanged; and corrupted payloads
were rejected. Replay and optimizer updates were zero.

Warm/fresh proposal transfer was measured rather than assumed. It was mixed
across seeds, so the promoted result is the control-flow and memory boundary,
not a claim that the scaffold prior reliably improves sample efficiency.

This is a narrow external loop/control-flow result. It does not prove that the
controller can synthesize arbitrary programs from scratch, that execution is
unbounded in deployment, or that the overall system has general continual
learning. The next pressure is structural synthesis beyond one-edit scaffold
adaptation, followed by integration with the amodal intention boundary.

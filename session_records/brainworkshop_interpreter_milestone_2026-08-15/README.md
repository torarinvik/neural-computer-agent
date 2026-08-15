# Interpreter milestone: mechanics verified, learning not started (2026-08-15)

Status: **diagnostic**. First step of
`docs/DECISION_CONTROLLER_IS_THE_INTERPRETER.md`. Nothing admitted, no bank
touched, no holdout spent.

## What was built

`interpreter_controller.py`: programs are data the runtime reads, executed one
microstep at a time under an explicit budget.

- an **instruction** is one row of program tensor; nothing enumerates
  instruction types;
- **operators are content-addressed**: the program carries a handle table, the
  controller emits an intention, and the runtime picks whichever handle that
  intention is nearest to;
- **workspace** is external slots beside the program;
- **budgets fail closed**: an exhausted tick records the status and emits
  nothing.

Execution is structural; only the choice of intention is learned. The module
runs in `teacher` mode, which reads the intention from the instruction and so
exercises the machinery with no trained network, and in `learned` mode, which
asks the controller.

## Results

| Check | Result |
| --- | --- |
| 1-back reproduced through interpretation | **1.000** over 447 ticks, every tick `halted` |
| Recorded lease value for the same capability | 1.000 |
| Operator table 6 → 7 rows | controller digest **unchanged**, behaviour still 1.000 |
| Workspace 1 → 16 slots | controller digest **unchanged**, behaviour still 1.000 |
| Microstep budget of 1 | status `budget_exhausted`, press `None` |
| Operand off the end of the workspace | status `invalid_operand`, press `None` |
| Untrained controller in `learned` mode | **0.501** |

The first row is the milestone: a capability the leases verified is now
produced by interpreting a program rather than by a Python path deciding the
action. Rows three and four are the invariants from the decision holding
mechanically rather than by promise — the operator vocabulary and the working
memory both live in the program, and neither resizes the controller.

The last row is the honest one. An untrained controller interprets at chance,
with ticks ending in all three statuses. Interpretation is a skill this
controller does not have.

## What is not done

- **The controller has not been pretrained to interpret.** This is the next
  chunk, and it follows the precedent of `controller_pretraining.py`: pretrain
  on generic mechanics with no task rule involved, freeze, then let programs
  carry capability.
- **`one_back_program` is an experimenter's reference program**, written to
  check that interpretation preserves behaviour. It is not learned and not
  admitted, exactly like the compiler in `counter_state_programs`.
- **No blueprint change yet.** The decision records that a two-way decoder
  cannot express an interpreter's micro-operations; this module sidesteps that
  with a separate small controller emitting into handle space rather than
  widening the curated one. Whether the curated controller is retrained or
  replaced is still open, and `AGENTS.md`'s weight-reset terms apply either
  way.
- **The old direct path is still live.** It should be marked legacy once
  interpretation covers what it covers.

## Cost, measured

| Path | Per environment tick |
| --- | ---: |
| Symbolic counter executor (Python) | 40 us |
| Frontend encode | 23 us |
| Neural controller, batch 1 | 335 us |

One controller forward costs `4.46 us` per item at batch 1 and `0.05 us` at
batch 512, an 89x difference that is dispatch overhead rather than arithmetic.
A 300-microstep tick at batch 1 would cost about 100 ms.

The consequence for optimisation: the interpreter is the expensive path and it
is tensor work, so batching across candidates and seeds is worth roughly two
orders of magnitude before any rewrite is worth discussing. The Python
symbolic executor is already 8x cheaper than the neural path it is being
compared against, and under this decision it is leaving the deployed path
entirely.

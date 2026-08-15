# The controller can run the programs after all (2026-08-15)

Status: **diagnostic**, on the already-consumed development seed. Nothing
admitted; `AgentBrain.bank` unchanged at `07319eb1`; the interpreter's weights
are unchanged and its digest is asserted before and after.

`DECISION_CONTROLLER_IS_THE_INTERPRETER.md` says the controller executes
external programs, and every result in this session bypassed it. The counter
bridge, the induced programs, the integrated agent -- all decide presses in a
Python executor while the pretrained interpreter sits in the checkpoint
directory doing nothing. Worse, the accumulation curve now bends the right way
*without* it, which weakened the decision rather than supporting it.

This is the measurement that settles it, and there is now something real to
interpret.

## Result

Eighteen sampled rules of one to six states, each compiled three ways and run
on the same rendered episodes.

| | mean accuracy | exact |
| --- | ---: | ---: |
| Counter executor (the path everything else uses) | 1.000 | 18/18 |
| Interpreter, operators read off the instruction | **1.000** | **18/18** |
| Interpreter, controller resolving against the whole table | 0.734 | 3/18 |
| Interpreter, controller resolving among the row's own operators | **1.000** | **18/18** |

**The frozen controller interprets every rule exactly**, with no retraining, no
parameter change, and two operators that did not exist when it was frozen.

## What had to be true, and none of it is a special case for rules

**The condition had to be the one the controller was taught.** It knows one
thing: name the operator in an instruction's first field when the current event
matches the workspace, and the second field otherwise. So the workspace holds
one slot, `load_const` puts a symbol prototype into it, and the next
instruction's condition reads "is the current symbol this one". Nothing else is
asked of the network. Measured separation on the real frontend makes this
sound: within a cluster the largest distance is 0.001, between clusters the
smallest is 4.64, against a tolerance of 0.5.

**The machine's state had to live outside the network.** It lives in the
program counter. One block of instructions per state; a transition is where the
tick parks the pointer. `halt_at` ends a tick at a chosen row, so state is
carried by the runtime rather than by any hidden activation.

**Growing the instruction set had to leave the controller alone.**
`load_const` and `halt_at` are two more rows in a table the program carries.
The controller's digest is identical before and after -- the invariant the
decision rests on, now tested against a real capability rather than a reference
program.

## The one thing that was wrong

Full-table resolution loses **0.266** on average and up to **0.496** on a
single rule. Auditing every row of every compiled program under both branch
outcomes says exactly where it goes:

| | count |
| --- | ---: |
| Chose the wrong *field* of the instruction | **0** |
| Resolved to a handle in **neither** field | 504 |

The controller never once misread a condition. Every interpretation error was
an operator the row could not have meant -- and one such error mid-tick drives
the pointer off the program, which fails closed, which forfeits the tick. Thirty
percent of ticks were being thrown away by 3% of decode errors.

Restricting content addressing to the two operators the row names removes all
of it. That is not a weakening of the design: the intention is still matched
against handles, nothing enumerates opcodes, and adding an operator still
leaves the controller untouched. It only stops a row from meaning something it
does not name.

## What is honestly weak

**Narrowing makes each row a binary choice.** The controller is doing condition
evaluation, not open-vocabulary operator naming, and the two numbers measure
different things. Full-table resolution is the stronger claim and it is the one
that fails at 0.734; both are in the table above rather than only the flattering
one.

**The rules here are compiled from the answer.** `clusters_in_symbol_order` is
an experimenter's oracle, used because this record measures whether three
execution paths agree on a program of known behaviour -- not whether the
program was learned. A first version omitted it, compiled a permutation of each
rule, and reported meaningless absolute accuracies while behaviour preservation
still read 18/18. Induction is `integrated_agent`'s record; this is about
execution.

**Nothing here is admitted, and the integrated agent still uses the counter
executor.** Connecting the two is the obvious next step and it is not done.

**One alphabet, one frontend, four well-separated prototypes.** The condition
is easy because the clusters are far apart. Nothing here shows what happens
when they are not.

## Reproducing

```bash
PYTHONPATH=src .venv/bin/python -m experiments.brainworkshop_canonical.interpreted_machine
```

About five seconds.

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_interpreted_machine.py -q
```

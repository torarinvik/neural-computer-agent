# Building on what it built (2026-08-15)

Status: **diagnostic**, five replicates on the already-consumed development
seed. Nothing admitted; `AgentBrain.bank` unchanged at `07319eb1`.

Read with `brainworkshop_compositional_transfer_2026-08-15`, which establishes
that a task the agent has never seen becomes cheap when its parts are in the
library. This asks the next question, and it is the one that separates
accumulating from interpolating: does a composite the agent *worked out itself*
become a part in its own right?

Nothing in the search knows about depth. It combines pairs of library records,
and a record is a record however it got there. So if depth appears, it appears
because admission made it available.

## Result

Four primitives, four pairs of them, then four triples -- each triple built as
`(a . b) . c` where `a . b` is one of the pairs in the stream, so a depth-2
route genuinely exists. Twenty triples across five replicates.

| | solved | composed | acquisition |
| --- | :--: | ---: | ---: |
| **pairs seen first** | 20/20 | **17** | **896** |
| pairs withheld | 20/20 | 9 | 2112 |
| control (no library) | 20/20 | 0 | 3648 |

**0.246 of a fresh agent.** And removing only the intermediate layer -- same
primitives, same triples, same seeds -- costs **0.58**, so about half the
advantage is the depth itself.

Nine of twenty triples were reachable from primitives alone, which is the
honest reason the withheld arm is not simply as expensive as the control: a
product sometimes collapses to something two primitives can already express.

## What is honestly weak

**The mean hides a real spread.** Per replicate the depth ratio is 0.31, 0.33,
0.40, 0.63 and **1.25**. In one world of five, having built the pairs made the
triples *more* expensive -- an admitted composite that a later task adopts and
then fails to confirm costs two full episodes and nothing recovers them.

**Depth is a side effect, not a plan.** The agent reaches depth 3 only because
a depth-2 result was admitted first. It never searches for a three-part
decomposition directly, so a triple whose pairs it never met is out of reach
unless a pair of primitives happens to cover it.

**One triple had eleven states**, which is past what the induction fallback can
represent at all -- so the control's "solved" on that task is an approximation
clearing the gate rather than an identification.

**Nothing is admitted, and the pool is four primitives over one alphabet.**

## Reproducing

```bash
PYTHONPATH=src .venv/bin/python -m experiments.brainworkshop_canonical.hierarchical_transfer
```

About forty seconds.

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_compositional_transfer.py -q
```

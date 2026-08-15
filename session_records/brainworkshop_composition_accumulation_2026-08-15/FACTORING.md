# The library finds its own parts (2026-08-15)

Status: **diagnostic**. Development seed 41, already consumed. Nothing is
admitted and `AgentBrain.bank` is checksummed before and after.

`README.md` in this directory showed a library paying off for the first time,
by trying every product of two library files under `and`, `or` and `xor`. Three
things were wrong with that as a mechanism. The combiners are a list an
experimenter wrote. The cost is `L^2 x combiners`. And the library stores whole
solved tasks, so it never holds anything smaller than a solution.

There is an exact alternative and it is sixty years old. Hartmanis and Stearns
(1966) showed a sequential machine decomposes into parallel components exactly
when its state set admits **closed partitions** -- a partition is closed when
knowing which block the machine is in determines which block it moves to, for
every input. Two closed partitions whose meet is the identity *are* a parallel
decomposition. Nothing is searched: the partitions are computed from the
transition function.

## Two changes

**The library holds parts.** Every induced machine is factored and its
components stored alongside it. A component carries transitions only -- closure
determines the next block, not the output -- so a part is a state-computer.

**Output is fitted, not enumerated.** Given two components, every observation
names one cell of a table indexed by (block, block, symbol), so the table is
read off the evidence in linear time. It expresses *any* output function of
that triple, of which `and`, `or` and `xor` are three.

Factoring works: 11 of the 14 composites decompose, and every decomposition
reconstructs the original behaviour exactly, checked by rebuilding the machine
rather than trusting the algebra. The three that do not factor collapsed under
minimisation and genuinely have no structure left.

## Result

Same curriculum, same ladder, same exact-1.000 bar on a held-out episode.

| Arm | Identified | Composite steps | Ratio | From library |
| --- | ---: | ---: | ---: | ---: |
| control -- induce every time | 9/18 | 18368 | 1.000 | 0 |
| products -- enumerate combiners | 9/18 | 16128 | 0.878 | 4 |
| **factored -- parts and fitted tables** | **11/18** | **14112** | **0.768** | 4 |

The four tasks the factored library solved cost **1120** labelled steps against
the control's **5376** -- **4.8x** -- and two of them are solved by *no other
arm*:

| Task | products | factored | control |
| --- | --- | --- | --- |
| 4-state composite | FAIL, 1792 | **fit:1+2, 448** | FAIL, 1792 |
| 4-state composite | FAIL, 1792 | **fit:1+2, 448** | FAIL, 1792 |

That is the result. Not a cheaper route to the same capability -- capability
neither the combiner library nor induction from scratch could reach at any rung
of the ladder. The same pair of parts explains both tasks, which is what having
a vocabulary is supposed to look like.

One more win is out of reach of the previous mechanism for a different reason:
`fit:-1+4` pairs a library part with the *trivial* one-state component, so it
is a single part plus a fitted table. There is no binary combination there for
a combiner list to express.

## Honest reading

**The arms are not nested.** The factored library keeps solutions *and* parts,
so it should contain everything the products library has, but deduplication
and ordering make the sets differ -- 8 entries against 9 -- and two tasks the
products arm won by combiner (`or:0+1`, `xor:5+6`) the factored arm re-induced
at 448 steps instead. The factored arm wins overall and on new capability, not
uniformly.

**Storing parts *instead of* solutions loses.** That was the first thing tried,
and it scored 1 library win against 4 and a worse ratio than the combiners:
nine whole machines collapse to three distinct components, and three components
cannot span the task space. Factoring extends a vocabulary; it does not replace
one. The record of that failure is why the arm now stores both.

**Five composites and two primitives are identified by nothing**, at any rung.
Seven composites defeat the control; factoring rescues two of them. The bar
here is exact 1.000 on a held-out 448-step episode, which is why totals look
low against records that used a 0.8 threshold.

## What this changes

The composition record left one question: whether the agent could find
factorisations without an experimenter enumerating products for it. It can, and
the mechanism is exact rather than heuristic -- closed partitions come out of
the transition function, and the output table comes out of the evidence.

That closes the loop the accumulation curve opened. Capability accumulates when
tasks share structure; the library discovers the shared structure itself; and
reuse costs a table fit rather than a search. What remains is that all of this
runs on machines induced by an experimenter's procedure, and the controller
still does not execute any of it.

## Reproducing

```bash
PYTHONPATH=src .venv/bin/python -m experiments.brainworkshop_canonical.composition_accumulation
```

About three minutes for all three arms.

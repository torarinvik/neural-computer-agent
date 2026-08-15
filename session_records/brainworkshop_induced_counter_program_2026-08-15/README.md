# Programs induced from feedback, not compiled from the answer (2026-08-15)

Status: **diagnostic**. Development seed 41, already consumed. Nothing is
admitted and `AgentBrain.bank` is checksummed before and after.

Two ceilings were sitting next to each other without touching. The counter
bridge showed a program family expressing **18/18** sampled rules, and was
called a ceiling because an experimenter's compiler read the rule and emitted
the answer. The identification ceiling showed **one episode of feedback
determines the task exactly** for nine of eighteen rules, and was called a
ceiling because the inferred machine was never handed to anything.

Joined, neither is a ceiling. The compiler is provenance-neutral code: it
turns *a Mealy machine* into a counter program and does not care where the
machine came from. Feeding it a machine inferred from the agent's own reward
is not an oracle.

## The pipeline

1. Cluster the frontend's own events -- the alphabet is discovered, not given.
2. Run one episode; invert its per-step reward into the target behaviour.
3. Infer the smallest Mealy machine consistent with that trace.
4. Compile it to a counter program.
5. Score it on a **different** episode.

Step 4 usually needs a map from the hypothesis's symbols to the executor's
input channels, and here there is nothing to map: the hypothesis was inferred
over cluster indices and the executor's channels *are* cluster indices. The
identity map is not a convenience, it is why no oracle is required.
`cluster_symbol_map` -- which renders a canonical frame per symbol to learn
what each symbol looks like -- is precisely the oracle step this replaces, and
it is not imported.

`tests/test_induced_counter_program.py` walks the function's syntax tree and
fails if it so much as names the rule, because "nobody looked" is a weaker
guarantee than "the code cannot see it".

## Result

| True states | Solved | Instructions | Counters | Accuracy |
| ---: | ---: | ---: | ---: | --- |
| 1 | 3/3 | 21-22 | 7 | 1.0000 |
| 2 | 3/3 | 43 | 9 | 0.9978, 1.0000, 1.0000 |
| 3 | 3/3 | 63-66 | 11 | 1.0000 |
| 4 | 1/3 | 81 | 13 | 1.0000 |
| 5 | 0/3 | - | - | - |
| 6 | 0/3 | - | - | - |
| **total** | **10/18** | | | **9 exactly correct** |

Every executed program halted inside its step budget on every tick. Instruction
and counter counts match the oracle-compiled ceiling exactly at each state
count, which is the check that induction changed the *provenance* and not the
program.

## Against the best searcher

| | solved | episodes | quality |
| --- | ---: | ---: | --- |
| Feedback-ranked search over the temporal family | 7/18 | 125 | 3 of 7 marginal, pooled 0.770-0.807 |
| Induced counter programs | **10/18** | **28** | **9 of 10 exactly 1.0000** |

Four rules are newly solved -- exactly the four the identification ceiling
flagged as *identified but inexpressible*, where the answer sat in the
evidence and the temporal family could not write it down. One rule is lost: a
four-state rule the searcher gated at 0.8058 by retrieving a near-miss, which
this pipeline declines to solve because it cannot identify it. Trading a
marginal gate for an honest abstention is the right direction.

**4.5x fewer episodes and 43% more rules**, and the wins stopped being
marginal.

## Where the constraint moved

Every one of the eight failures is an *identification* failure, not an
expressiveness one: the compiler would emit a correct program for any machine
handed to it, and no machine was found within budget. Rules of five and six
states are not identifiable from passive observation at any budget worth
spending -- Gold's NP-hardness, measured in the previous record.

So the sequence over this benchmark now reads:

1. expressiveness was the constraint -- the counter bridge answered it;
2. search was the constraint -- dedup and feedback ranking answered it, 11.5x;
3. expressiveness was the constraint again -- this record answers it, 4.5x;
4. **identification is the constraint**, and passive observation cannot beat it.

The next move is the one Angluin's L* makes: stop watching and start asking.
The agent currently runs whatever program is installed and reads what comes
back. Choosing an action sequence *in order to disambiguate* between
hypotheses is what turns an NP-hard passive problem into a polynomial active
one, and nothing in this architecture does it yet.

## What is still not claimed

Nothing here is admitted, and the bank is unchanged. The induced programs are
not in the library, do not compose, and the accumulation question is untouched
by this record -- these are single-task inductions, not accumulated capability.
The controller does not run in this path either; presses come from the counter
executor, exactly as in the original bridge. This closes the provenance gap
that made the bridge a ceiling, and it closes nothing else.

## Reproducing

```bash
PYTHONPATH=src .venv/bin/python -m experiments.brainworkshop_canonical.induced_counter_program
```

About twelve seconds.

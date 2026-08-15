# One episode of feedback already determines half the benchmark (2026-08-15)

Status: **diagnostic**. Development seed 41, already consumed. An
experimenter's inference procedure, run to establish what the evidence
supports. Nothing is admitted, no inferred machine is given to the searcher,
and `AgentBrain.bank` is checksummed before and after.

Three quantities decide whether a learner is any good and this repository had
measured two. *Expressiveness*: 18/18 for the counter family against 7/18 for
the temporal one. *Search cost*: 1432 verifier episodes down to 125. Missing
was *identifiability* -- how much evidence it takes to pin the task down at
all -- without which there is no way to say whether 125 episodes is close to
optimal or absurd.

## Method

Only what the agent already has. Symbols come from clustering its own
frontend's events, so the alphabet is discovered rather than supplied; all
eighteen rules yield exactly 4 clusters. Outputs come from inverting one
scored episode's per-step reward, `target[t] = action[t] if reward[t] else
1 - action[t]`.

From that single trace, the smallest Mealy machine consistent with it. Scored
by predicting a *different* episode, not by comparing to the generating
machine: cluster and state names are arbitrary, so a digest comparison would
flatter or punish the result for the wrong reason.

## Result

| True states | Identified | Predicting a held-out episode exactly |
| ---: | ---: | ---: |
| 1 | 3/3 | 3 |
| 2 | 3/3 | 2 |
| 3 | 3/3 | 3 |
| 4 | 1/3 | 1 |
| 5 | 0/3 | 0 |
| 6 | 0/3 | 0 |
| **total** | **10/18** | **9** |

**Nine of eighteen rules are recovered exactly from a single episode**, in
14.5 seconds of inference. The inferred state count equals the true minimal
state count in every case. One two-state rule is identified but predicts the
held-out episode at 0.9978: consistent with the probe, under-determined by it.

## Against what the searcher managed

The feedback proposer gates 7 of 18 and spends 5 to 9 episodes doing it.
Crossing the two:

| | count |
| --- | ---: |
| Exactly identified **and** solved | 5 |
| Exactly identified, **not** solved | **4** |
| Solved, **not** identified | 2 |

The four in the middle row are the finding. For those rules a single episode
of feedback determined the task completely, and the program family could not
express the answer. They are not near misses either: the best achievable
agreement across the entire proposal space is 0.7121 to 0.7701, on tasks the
evidence pins down exactly. That is not a search failure, and no proposer can
fix it.

The two in the last row are worth as much. Both are `retrieve` wins sitting on
the gate -- best agreements 0.8036 and 0.8058 against a 0.8 threshold. One of
them *is* identified, at 0.9978 rather than exactly; the other is not
identified at all. Either way the searcher accepted a program that clears the
gate and is not the rule. A gate passed is not a task learned, and these are
the same two marginal retrievals the first curve already flagged.

## Where identification stops, and why that matters

Five- and six-state rules are not identified, and the boundary is sharp rather
than gradual: four states finishes in well under a second, five does not
finish under a budget twenty times larger. Eight episodes instead of one does
not help, which rules out thin evidence as the explanation.

This is Gold's 1978 result -- inferring a minimal automaton from a passive
sample is NP-hard -- met in practice. It is also the argument for the next
step. Angluin's L* buys polynomial guarantees by asking *active* queries
rather than observing harder, and the agent currently only observes: it runs
whatever program is installed and reads what comes back. Choosing an action
sequence *in order to disambiguate* is the move that turns an intractable
problem into a tractable one, and nothing in this architecture does it yet.

## The obvious tool that does not work here

Greedy state merging -- RPNI, the standard method -- is kept in the module and
reports **31 states for a two-state rule**. A single episode is a chain: every
node has exactly one outgoing edge, so the earliest merges have no evidence
against them, are accepted, and poison every later one. This is recorded
because it is the first thing a reader will reach for, and
`tests/test_identification_ceiling.py` asserts the failure so it stays
recorded.

## What this changes

The three-way separation is now complete, and the ordering is not what the
last week of work assumed:

1. **Information** is not the constraint. One episode fully determines half
   the benchmark.
2. **Search** is no longer the constraint. It was, and it was fixed.
3. **Expressiveness** is the constraint, on four rules where the answer was
   sitting in the evidence and could not be written down.

The remaining work is not a better proposer over this program family. It is a
program family that can express what one episode already reveals -- and, past
four states, an agent that asks rather than watches.

## Reproducing

```bash
PYTHONPATH=src .venv/bin/python -m experiments.brainworkshop_canonical.identification_ceiling
```

About fifteen seconds.

# 18/18, and the constraint was the shape of the evidence (2026-08-15)

Status: **diagnostic**. Development seed 41. Nothing admitted,
`AgentBrain.bank` checksummed before and after every run.

The previous record ended by pointing at Angluin's L*: stop watching, start
asking. Both halves of that were wrong, and the way they were wrong is the
result.

## Active querying is not available here

`RenderedBrainWorkshopVerifier` builds its entire symbol stream from a seed in
its constructor, and `score` reads the stream at the current position without
ever consulting the action. The agent chooses presses, not stimuli, so there
is no membership query to make. And because inverting the reward reveals the
target label at every eligible step *whatever the agent pressed*, its actions
carry no information-gathering value either. Active learning has nothing to
grip on in this environment.

## Neither did a better solver

If the five-state wall were computational, a stronger algorithm on the same
data should break it. EDSM (Lang, Pearlmutter and Price, 1998 -- the Abbadingo
winner) is the obvious candidate: RPNI takes the *first* consistent merge,
which on a near-chain is an evidence-free guess, while EDSM scores every merge
by how much agreement its determinizing cascade produced and takes the best.

On one long episode it does no better -- 56 states for a five-state rule,
against RPNI's 52. The reason is structural and it is the same reason RPNI
fails: a random exogenous stream produces a prefix tree that is a bundle of
*chains*. Episodes share almost no prefixes, so there are no overlapping
transitions, so there is no evidence to drive anything.

## The constraint was segmentation

Same rule, same total number of labelled steps, different episode lengths:

| 1792 steps split as | 4 states | 5 states | 6 states |
| --- | --- | --- | --- |
| 1 x 1792 | exact, 1.000 | none | none |
| 4 x 448 | none | none | none |
| 28 x 64 | 1.000 | 1.000 | 0.846 |
| 112 x 16 | **1.000** | **1.000** | **1.000** |

Many short episodes make the prefix tree branch, which is exactly what merging
needs. The two methods turn out to be complements rather than rivals -- exact
search wins on few long episodes at small state counts, merging wins on many
short ones -- so `infer_machine` now tries both, and accepts a merged result
only if it stayed inside the same complexity bound the exact search uses. A
"machine" with one state per few observations has explained nothing.

## End to end, through the real pipeline

Induce from feedback, compile to a counter program, execute against the
verifier on a held-out 448-step episode. The learning budget is measured in
labelled steps so the comparison is matched:

| Learning feedback | Steps per rule | Solved | Exactly 1.000 |
| --- | ---: | ---: | ---: |
| 1 episode x 448 | 448 | 10/18 | 9 |
| **28 episodes x 16** | **448** | **15/18** | 13 |
| 112 episodes x 16 | 1792 | **18/18** | 14 |

The middle row is the finding: **identical feedback, 448 labelled steps either
way, and cutting it into 28 short episodes solves five more rules.** Four times
that budget, still in short episodes, solves all eighteen.

Every induced program halted inside its step budget on every tick. Instruction
counts run 21 to 130, matching the oracle-compiled ceiling at every state
count -- induction changed the provenance, not the program. The four rules
below 1.000 land at 0.9799 to 0.9978.

## What this actually says

The benchmark's 448-step episodes are close to the worst possible shape for
structure learning, and three records in a row blamed the agent for it. The
sequence over this benchmark now reads:

1. expressiveness -- answered by the counter bridge;
2. search -- answered by dedup and feedback ranking, 11.5x;
3. expressiveness again -- answered by induction, 4.5x;
4. identification -- **answered by shortening episodes**, 10/18 to 18/18.

Episode length is an environment parameter, not an agent capability, and this
agent cannot reset the world. So the honest conclusion is uncomfortable: the
last barrier on this benchmark fell to a change in how experience is
*delivered*, not to anything the architecture learned to do. An agent that
could choose when to reset would have this lever itself, and that -- not
active querying, which this environment forbids -- is the capability the
result argues for.

## What is still not claimed

Nothing is admitted, nothing composes, the accumulation question is untouched,
and the controller does not run in this path. This closes identification on
this benchmark and closes nothing else.

## Reproducing

```bash
PYTHONPATH=src .venv/bin/python -m experiments.brainworkshop_canonical.induced_counter_program --learning-episodes 28 --learning-steps 16
```

About thirteen seconds; the 112-episode variant takes twenty-two.

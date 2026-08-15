# The accumulation curve exists, and it bends the wrong way (2026-08-15)

Status: **diagnostic**. Development seed 41, already consumed. The curriculum
grows a scratch copy of the library; `AgentBrain.bank` is checksummed before
and after and was not written.

`DECISION_CONTROLLER_IS_THE_INTERPRETER.md` named this curve as its own
falsifier and closed with "that curve does not exist yet." It does now.

## What was measured

Eighteen sampled rules -- the same population the baseline and both ceilings
were measured on -- arrive one at a time in ascending complexity order. The
searcher runs against whatever library exists at that moment. Two arms differ
in one bit:

- **growing** -- a gated winner is admitted, so rule N faces everything learned
  from rules 1..N-1;
- **control** -- the library is restored before every rule, so each rule is
  learned by a fresh agent.

Cost is verifier episodes spent before a rule gates, including the one
re-derivation and two confirmation episodes admission requires.

## The headline

| | growing | control |
| --- | ---: | ---: |
| Rules gated | 7 / 18 | 7 / 18 |
| Rules reproducing on unseen episodes | 7 | 7 |
| Verifier episodes spent | **1432** | **886** |

**Cost ratio 1.616.** A growing library made the curriculum 62% *more*
expensive and solved not one additional rule. The library grew from 3 files to
7 and changed no outcome.

## Reuse is real, and it is only ever caching

Three rules were solved by retrieving a file learned earlier in the same
curriculum, and the saving is large:

| Position | States | growing | control | ratio | growing winner | control winner |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 3 | 2 | 7 | 34 | **0.21** | `retrieve:4` | `invent:[~1+3]` |
| 7 | 3 | 8 | 18 | **0.44** | `retrieve:5` | `invent:[1+2]` |
| 11 | 4 | 8 | 18 | **0.44** | `retrieve:5` | `invent:[1+2]` |

At all three the retrieved file and the independently invented one score
*identically* on the confirmation episodes (0.797, 0.770, 0.807). Retrieval is
not winning with a different, worse program -- it is short-circuiting the
re-invention of the same one, at a quarter to a fifth of the cost. That is
genuine reuse.

But it is the weakest kind. Every winner in the growing arm is either a
`retrieve` of an exact behavioural duplicate or a fresh `invent`. Across the
whole curriculum there were **zero composes, zero inverts of a learned file,
and zero ANDs over a learned file**. The library never combined its parts to
reach anything it could not already do. Capability transfers only when the new
task happens to be the old task.

## Why the total still lost: the searcher pays for the library by the item

Proposals are enumerated in a fixed order and every executable one is run
against the verifier until something gates. Growing the library from 3 files to
7 grew the proposal list from 94 to 190, and the prefix a rule must walk before
reaching the first fresh invention from **18 to 69**.

Eleven of the eighteen rules are outside what this family can express -- the
expressiveness record already established that -- so they gate on nothing and
execute the entire list:

- unsolvable rule, growing arm: **120** episodes
- unsolvable rule, control arm: **67** episodes

The tax is paid 11 times and the savings collected 3 times. That is the whole
1.616, and it is arithmetic, not noise.

## What is not claimed

The three retrieval wins all sit close to the 0.8 gate, and closeness to a gate
is exactly what this repository's lease machinery exists to be sceptical about.
Pooling each winner's confirmation episodes and asking whether its true rate can
be ruled out as sitting at or above the gate:

| Position | pooled | P(true rate >= 0.8) | ruled out at 1%? |
| ---: | ---: | ---: | --- |
| 3 | 0.797 | 0.42 | no |
| 7 | 0.770 | 0.015 | **no** |
| 11 | 0.807 | 0.71 | no |

None is ruled out at alpha = 0.01, so these are not being called fake. They are
also not being called solid: position 7 is the thinnest, at 0.770 against a
never-press baseline of 0.724 on that rule, and 0.015 misses the bar without
much room. The four fresh inventions are a different matter -- 0.865 to 1.000,
none marginal.

Also not claimed: that a bank limitation caused any of this. Every gated winner
was storable, and `solved_winners_the_bank_cannot_hold` is 0.

## What this changes

The falsifier was stated as "if interpretation does not bend the accumulation
curve, the external-program story is decoration." The curve is now measured for
the *pre-interpreter* path, and it establishes the baseline that any
interpreter work has to beat: **1.616, with reuse confined to exact duplicates.**

The result also localises the problem, and not where the last three days of
work assumed it was. The bottleneck is not representation -- the counter
bridge closed that at 18/18. It is not the controller -- the controller never
ran differently between the two arms. It is that **the searcher has no way to
decide what to try**, so every file the library accumulates becomes another
item to execute blindly. Under a fixed enumeration order, a library is a
liability that occasionally pays off.

This is a direct argument for the proposer that chooses by expected
information, and against doing more interpreter work first: an interpreter
makes programs longer and the search space larger, and there is now a measured
curve saying this search cannot afford the library it already has.

## Reproducing

```bash
PYTHONPATH=src .venv/bin/python -m experiments.brainworkshop_canonical.accumulation_curve
```

About six minutes.

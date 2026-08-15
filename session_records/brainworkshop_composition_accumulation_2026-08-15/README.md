# The library composes, once the tasks have parts to share (2026-08-15)

> **Superseded by `brainworkshop_adversarial_audit_2026-08-15/`.** A bug in
> `rule_automata.minimize` was changing the behaviour of 42.5% of machines and
> suppressed the numbers below. Post-fix the library arm reaches 18/18 at a
> cost ratio of 0.141, robust across four primitive pools, and the
> factorisation advantage recorded here is withdrawn.

Status: **diagnostic**. Development seed 41, already consumed. Nothing is
admitted and `AgentBrain.bank` is checksummed before and after.

Three arms of the accumulation curve reported the same thing: zero composes,
zero inverts of a learned file, zero ANDs over a learned file. Each time that
was read as a fact about the architecture. It was a fact about the sampler.

`rule_automata.sample_rule_population` draws every rule independently. Across
the eighteen sampled rules, the largest pairwise agreement between any two --
or its complement -- is **0.808**, and that is between a one-state rule and a
four-state one, which is coincidence rather than structure. Independent
samples share nothing, so a library can only ever help by already containing
the answer. That is exactly the reuse the curve measured, and it is the only
reuse that distribution permits. **Accumulation was impossible by
construction, and three records blamed the agent for it.**

## A distribution where the question has an answer

`compositional_rules` builds tasks mechanically, not by hand: primitives are
sampled once into a shared pool, and a task is the *product* of two primitives
under a boolean combiner. Products of Mealy machines are Mealy machines, so
nothing already written needs changing, and a composite of two three-state
primitives is genuinely harder -- up to nine states -- but decomposable.

The structure is real and measurable: composites sharing a primitive agree
with each other at **0.633** on a long stream, against **0.553** for
composites that share nothing.

## Protocol

Curriculum: 4 primitives, then 14 composites. Both arms climb the same
evidence ladder -- 7 short episodes of 16 steps, then 14, 28, 56, 112 --
stopping when a hypothesis reproduces a held-out 448-step episode **exactly**.
They differ in one thing: the growing arm first checks whether a library
machine, or a product of two library machines under a combiner, already
explains the evidence. Checking costs no verifier evidence, so any difference
is evidence the library carried.

Cost is labelled steps, because the arms stop at different rungs.

## Result

| | growing | control |
| --- | ---: | ---: |
| Identified | 9/18 | 9/18 |
| Labelled steps on composites | **16128** | 18368 |
| Cost ratio | **0.878** | |
| Composites solved from the library | **4** | 0 |

And on the tasks where the library actually fired:

| | growing | control | ratio |
| --- | ---: | ---: | ---: |
| The 4 library wins | **448** | 2688 | **0.167** |
| The 7 composites identified in both arms | 3584 | 5824 | **0.615** |

**6x cheaper where the library applies, 1.6x cheaper across everything both
arms could learn.** The overall 0.878 is diluted by seven composites that
failed in both arms and burned the full ladder each time.

The labels matter as much as the numbers: `and:0+1`, `or:0+1`, `xor:0+1`,
`xor:5+6`. Every one is a **product of two library machines** -- not a
retrieval. Across three arms of the original curve there was not a single
compose. Here there are no retrievals at all, only composes.

## What is not being claimed

**The winning factorisations are not the generating ones.** `parts_in_library`
is zero for every library win: the primitives those composites were actually
built from were never in the library, because only 2 of the 4 primitives were
identified. The library found *different* decompositions out of machines it had
induced from earlier composites, which reproduce the held-out episode exactly.
That is real composition and it is not recovery of the intended parts, and the
two should not be confused.

**The identification counts are lower than the induced-program record because
the bar here is stricter.** A hypothesis must reproduce the held-out episode at
exactly 1.000; the 0.996 to 0.998 results that counted as solved against a 0.8
threshold count as failures here. That makes 9/18 look weak and makes the
composition claim strong, since only exact matches can win.

Seven composites were identified by neither arm within 1792 labelled steps.

## What this changes

The accumulation question has an answer for the first time, and it is not the
one three records implied. A library of induced machines does compose, does
make later tasks cheaper, and the effect is large where it applies. What was
missing was never the mechanism; it was a task distribution in which sharing
was possible.

This also retires the reading that the external-program story is decoration.
On i.i.d. tasks it is decoration -- necessarily, for any architecture. On
tasks with shared structure the same library is worth 6x.

The open question moves accordingly: the effect here comes from an
experimenter's enumeration of products over a small library, which is
tractable at nine files and is `L^2 x combiners` at scale. What earns the
next measurement is whether the *agent* can find such factorisations without
enumerating them.

## Reproducing

```bash
PYTHONPATH=src .venv/bin/python -m experiments.brainworkshop_canonical.composition_accumulation
```

About eleven minutes.

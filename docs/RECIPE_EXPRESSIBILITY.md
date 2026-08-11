# Recipe expressibility boundary

The exported games session reports a basis of opaque operations over abstract
slots and one family that performs a simultaneous two-slot effect. That
quantitative result is **SINGLE-SOURCE, UNREPLICATED**. It is a useful
hypothesis, not evidence that the canonical controller has learned a missing
operation.

## Architectural decision

The minimal long-term extension should be structural composition, not a
benchmark-named `PAIR_FLIP` instruction. The candidate is a generic
`PARALLEL(left, right)` combinator:

- each child is an existing local instruction;
- both children read the same pre-step register;
- their write sets must be disjoint;
- both effects commit as one verifier-visible step.

This closes the class of simultaneous independent local effects. A two-valued
`INC` is a bit flip, so `PARALLEL(INC i, INC j)` is the pair-flip
specialization without putting “flip” into the architecture.

The ABI is implemented as `neural_computer.RecipeBasis` with an explicit
`one_instruction_one_verifier_step_v1` atomicity contract. It is an instrument
for the external computation basis, not a new controller branch or a task
solver.

## Current mechanical result

On six slots with eight values, the baseline contains `118` atomic candidates.
The generic parallel extension contains `5,518` candidates, a `46.8x` larger
exhaustive search space. The paired-increment target is correctly reported as
`inexpressible` by the baseline after all `118` candidates are checked and is
represented by the extension after `119` checks. This establishes the
diagnostic boundary and exposes the search-cost tradeoff; it is not yet a
learned-interpreter result.

## Required learned audit before promotion

The next experiment must train one fixed interpreter only on random programs
over the generic basis, with no task-family data in its weights. At two seeds,
it must report:

1. held-out accuracy on unseen baseline programs;
2. held-out accuracy on double-length baseline programs;
3. the same two curves after adding `PARALLEL` to the training grammar;
4. search proposals or expansions per found recipe;
5. a held-out paired-effect target and the old-basis negative control;
6. exact retention of all earlier interpreter behavior after the extension.

The search must fail closed. “Budget exhausted” is not equivalent to
“inexpressible”: a target is outside the basis only when an exhaustive or
otherwise certified reachable-effect check proves no atomic representation.
If the target is expressible but the learned proposal misses it, that is a
search or proposal-distribution failure and must be recorded separately.

No promotion should occur until the richer basis's interpreter cost is
measured against the baseline at matched verifier bits and the extension wins
on expressibility without a material loss on unseen baseline execution.

The first paired two-seed calibration at 500 updates was correctly rejected as
undertrained: the parallel target reached `0.8506`/`0.9805`, but old-basis
length-two accuracy was only `0.5830`/`0.5264` in the extension arms and the
length-four curve was lower. The durable result is the harness diagnosis, not
the score. Evidence is archived in
`session_records/recipe_interpreter_undertrained_calibration_rejected_2026-08-11/`.

## Narrow promoted result

At the registered 2,500-update rung, the generic parallel arm reached stable
`>=0.9` old-basis length-two and double-length execution by `1,500` updates on
both seeds. The atomic baseline reached the same old-basis threshold at `2,000`
updates on seed `70422` and did not reach it on seed `70421`. The paired target
was stably learned by both extended arms. This promotes the bounded generic
composition mechanism under random-program pretraining, with the explicit
confound that the richer arm also practices a richer program distribution.
Evidence is archived in
`session_records/recipe_parallel_composition_promoted_2026-08-11/`.

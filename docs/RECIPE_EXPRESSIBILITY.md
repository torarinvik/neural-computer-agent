# Recipe expressibility boundary

The exported games session reports a basis of opaque operations over abstract
slots and one family that performs a simultaneous two-slot effect. That
quantitative result is **SINGLE-SOURCE, UNREPLICATED**. It is a useful
hypothesis, not evidence that the canonical controller has learned a missing
operation.

## Architectural decision

The first required boundary is an explicit arithmetic modulus. Every
increment/decrement instruction now carries its modulus as data, and the
runtime validates it against the target slot's observed value domain. A
single global `VALUES=8` is not a valid substitute: it makes an increment on a
two-valued slot correct only when that slot currently holds zero.

The separate structural-composition candidate remains a generic
`PARALLEL(left, right)` combinator, not a benchmark-named `PAIR_FLIP`
instruction:

- each child is an existing local instruction;
- both children read the same pre-step register;
- their write sets must be disjoint;
- both effects commit as one verifier-visible step.

This closes the class of simultaneous independent local effects. A two-valued
`INC` is already a bit flip, so a two-valued pair toggle is expressible as
`INC(i, m=2); INC(j, m=2)` using existing instructions. `PARALLEL` is only
needed when those effects must commit as one verifier-visible atomic step.

The ABI is implemented as `neural_computer.RecipeBasis` with an explicit
`one_instruction_one_verifier_step_v1` atomicity contract. It is an instrument
for the external computation basis, not a new controller branch or a task
solver.

## Corrected modulus result

For slot domains `(2, 2, 8, 8, 8, 8)`, applying the legacy global modulus of
eight to one increment matches the correct family transition at per-slot
rates `[0.5, 0.5, 1.0, 1.0, 1.0, 1.0]`. The explicit per-slot modulus makes
the same transition exact. This is the structural defect behind the earlier
misreading of the two-valued toggle result. The `RecipeInstruction` ABI now
uses `(op, i, j, m)` for arithmetic operations, and `RecipeBasis` accepts
per-slot value domains.

## Separate atomic-composition result

On six slots with eight values, the baseline contains `118` atomic candidates.
The generic parallel extension contains `5,518` candidates, a `46.8x` larger
exhaustive search space. The simultaneous two-slot increment target is still
correctly reported as `inexpressible` by the one-instruction atomic baseline
and represented by the parallel extension. That is an atomicity result, not
evidence of a missing toggle primitive and not evidence that sequential toggle
composition was impossible. The prior interpreter reports are therefore
historical artifacts with corrected interpretation status, not current
promotion records.

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
on expressibility without a material loss on unseen baseline execution. The
first fixed-slot modulus result was insufficient because a wrong-modulus
control could still succeed by memorizing slot identity; domain assignments
must be randomized.

## Narrow learned modulus result

The corrected two-seed, 1,500-update mixed-domain audit now includes dedicated
single-increment probes. Across both seeds and both atomic-only/parallel
training arms, `m=2` and `m=8` reach stable exact execution by at most `1,500`
updates, with zero replay and `192,000` unique random-program steps per arm.
This promotes the explicit modulus contract as a narrow learned arithmetic
capability. The simultaneous parallel target remains unstable and is not
promoted. Evidence is archived in
`session_records/recipe_modulus_learned_narrow_promotion_2026-08-11/`.

The first paired two-seed calibration at 500 updates was correctly rejected as
undertrained: the parallel target reached `0.8506`/`0.9805`, but old-basis
length-two accuracy was only `0.5830`/`0.5264` in the extension arms and the
length-four curve was lower. The durable result is the harness diagnosis, not
the score. Evidence is archived in
`session_records/recipe_interpreter_undertrained_calibration_rejected_2026-08-11/`.

## Superseded prior promotion

The earlier 2,500-update parallel-composition report used a uniform modulus-8
world and was described as if it validated a two-valued toggle gap. That
interpretation is withdrawn. The run remains a valid historical atomicity/
composition diagnostic, but it is not evidence about the corrected
mixed-domain modulus problem.
Evidence is archived in
`session_records/recipe_parallel_composition_promoted_2026-08-11/`.

## Causal modulus promotion

The corrected 1,500-update, two-seed audit randomizes the `(2, 2, 8, 8, 8,
8)` domain assignment for every random program. Across both atomic-only and
parallel-training arms, the correct `m=2` and `m=8` probes reach stable exact
execution by update `300`. The byte-identical wrong-`m=8` probe for a
two-valued target finishes near `0.5` on every arm, as expected when the
modulus operand is used rather than ignored. This is the strongest current
learned result: a narrow causal modulus capability, not general continual
learning. Evidence is archived in
`session_records/recipe_modulus_randomized_causal_promotion_2026-08-11/`.

The earlier fixed-slot modulus record remains a valid execution result but is
superseded as causal evidence by this randomized-domain control:
`session_records/recipe_modulus_learned_narrow_promotion_2026-08-11/`.

## Arithmetic-family promotion

The follow-up two-seed, 1,500-update audit extends the randomized-domain
control to all arithmetic forms: `INC`, `DEC`, `CINC`, and `CDEC`, each at
`m=2` and `m=8`. All eight correct probes reach stable `>=0.9` execution in
all four arms by the end of the rung. The wrong-`m=8` two-valued control still
finishes near `0.5` in every arm. This promotes explicit modulus use across
the arithmetic family, not merely one increment template. Evidence is
archived in
`session_records/recipe_arithmetic_family_causal_promotion_2026-08-11/`.

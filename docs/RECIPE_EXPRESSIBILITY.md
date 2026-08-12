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

## Held-out composition promotion

The parallel-training arm now excludes the exact target
`PARALLEL(INC(0,m=2), INC(1,m=2))` from its random-program stream, including
the reversed child order. At 1,500 updates, both parallel arms reach stable
exact target execution by update `300`, while both atomic-only arms remain at
zero. This promotes narrow held-out parallel composition, not a claim that
the richer arm learns the old basis faster: its old-basis curves remain a
separate training-distribution comparison. Evidence is archived in
`session_records/recipe_parallel_heldout_causal_promotion_2026-08-11/`.

## Bounded sequence compiler boundary

The recipe ABI now distinguishes atomic expressibility from finite sequence
expressibility. `RecipeBasis.sequence_probe()` performs a breadth-first search
over opaque register effects, merging equivalent prefixes and returning the
shortest sequence found within a configured bound. It therefore finds the
two-valued toggle as `INC(0, m=2); INC(1, m=2)` without adding a toggle or pair
primitive.

The result is fail-closed: `inexpressible` means the complete finite bound was
checked, while `budget_exhausted` means that search stopped before the bound
was certified. This is an execution/compiler foundation only. It does not yet
show that the frozen controller can learn arbitrary recipes, discover useful
instruction sequences efficiently from Brain Workshop outcomes, or grow an
unbounded program library. The next audit must compare this compiler against
stochastic proposal search under matched verifier bits, then test retention and
held-out transfer when a discovered sequence is stored as an external file.

## Generic external control flow

The external computation boundary now has a separate versioned counter-machine
ABI with `INC`, `DEC`, unconditional jump, zero/nonzero conditional jump, and
`HALT`. Two non-negative counters plus zero-tested control flow provide the
standard unbounded counter-machine foundation for general computation, while
every deployed execution remains fail-closed under an explicit step budget and
counter limit. This is a computational substrate, not a hand-written task
solver or a claim that the controller has learned universal program induction.

The `ControlFlowOutcomeSearch` proposal layer updates only aggregate scalar
outcome credit and keeps program files in `ControlFlowProgramMemory`. It does
not persist verifier rows or expose target program names. The four-seed audit
started from one protected transfer file and acquired a second transfer file
whose loop increments a different opaque counter. Both forward and reversed
input orders reached `1.0000` held-out execution, retained the source file,
reloaded exactly, rejected shuffled feedback and corrupted payloads, and
performed no replay or optimizer updates. The experiment intentionally keeps
the source scaffold fixed; this promotes a narrow loop/control-flow and
external-file growth boundary, not unrestricted program induction or general
continual learning. Evidence is archived in
`session_records/recipe_control_flow_growth_promoted_2026-08-12/`.

## Bounded from-scratch control-flow induction

The next audit removes the executable source scaffold. A generic finite
enumerator searches all four-instruction programs over the two-counter basis,
while the verifier privately checks a loop that clears an opaque counter. The
search receives only scalar exact-match outcomes and admits an equivalent
program into the external file memory; it does not need to recover the
verifier's canonical instruction sequence. Across four seeds and both input
orders it reaches `1.0000` held-out and reload accuracy, retains a protected
straight-line source file, and rejects missing evidence, shuffled feedback,
and corruption. A ten-candidate cut-off reports `budget_exhausted`, proving
that a budget stop is not mislabeled as `inexpressible`.

This is the first bounded from-scratch loop-induction result, but it is still
finite enumeration over one short program length. It does not establish
efficient arbitrary program synthesis, unrestricted execution, or general
continual learning. Evidence is archived in
`session_records/recipe_control_flow_induction_promoted_2026-08-12/`.

## Outcome-only external recipe files and scope isolation

The recipe basis now has a versioned external-file bridge:
`OutcomeOnlyRecipeSequenceSearch` proposes generic sequence edits and updates
only aggregate scalar-credit statistics, while `ExternalRecipeProgramMemory`
admits candidates through a stable verifier prefix and persists protected files
with checksums. Candidate-evaluation history is scoped by an opaque external
binding key, so rejecting a candidate in one context cannot prevent a later
context from evaluating the same program. The global edit prior may transfer;
raw verifier rows and task labels do not.

The two-seed order-sensitive growth audit passed source and two auxiliary-file
retention, target held-out mastery, wrong-order rejection, shuffled-feedback
rejection, reload, checksum, and zero-replay gates. It did **not** promote
scalar search-prior transfer: after two auxiliary acquisitions, target search
used `74` versus `28` proposals on seed `17` and `34` versus `23` on seed `18`
for the warm versus fresh controls. The negative result is important: a shared
operator prior can be transferred safely, but it is not yet a reusable
sample-efficiency gain. Evidence is archived under
`session_records/recipe_outcome_only_sequence_growth_rejected_2026-08-12/`.

The next pressure is context-conditioned proposal credit over instruction
content and position, with an explicit exploration floor. More external slots
or longer programs are not justified until that proposal policy improves a
held-out learning curve without sacrificing the scope-isolation and retention
gates.

## Context-conditioned proposal credit

`OpaqueContextRecipeProposalMemory` adds a replaceable external policy layer
above the sequence search. It stores aggregate scalar quality keyed by an
opaque context and a content-addressed candidate digest; it stores no verifier
rows, task labels, or controller updates. Candidate-history scope remains
separate from context, so a new lifetime can reevaluate a previously seen
candidate while retaining the old contextual credit. A nonzero exploration
floor keeps every candidate reachable, including in an unseen context.

The two-seed audit acquired two contradictory order-sensitive recipes in two
opaque contexts, persisted the policy, then reacquired both in fresh lifetimes
without replay. Both contexts retained `1.0000` held-out accuracy. Warm
proposal counts were `1` versus `9` and `2` versus `17` on seed `17`, and `3`
versus `17` and `2` versus `11` on seed `18`, for warm/fresh ratios of
`0.1111`, `0.1176`, `0.1765`, and `0.1818`. The unseen-context distribution
was unbiased, each trained context preferred its own recipe, shuffled feedback
was rejected, and policy reload was exact. Evidence is archived under
`session_records/recipe_context_conditioned_proposal_credit_promoted_2026-08-12/`.

This promotes bounded replay-free contextual proposal reuse only. The policy
currently credits an exact whole candidate digest; it does not yet factorize
instruction identity, insertion position, or reusable sub-sequences across
related contexts, and it does not establish general continual learning or
unrestricted memory growth.

## Factorized instruction/position proposal credit

`RecipeProgramProposalFactors` makes the generic edit ABI explicit: an
operator, one or two opaque positions, and content-addressed instruction
digests. `FactorizedOpaqueContextRecipeProposalMemory` stores aggregate scalar
quality over those factors in external shared and context-local tables. It
does not retain whole candidate rows; a shared factor prior transfers useful
edits to a different parent program, while local evidence disables that prior
for a known context and can learn a reversal. The exploration floor remains
active for every candidate.

The four-seed audit used mixed domains `(2, 8)`, acquired `INC(0,m=2)` plus
`CINC(1|0,m=8)` in one context, then transferred the same opaque insertion
instruction/position factors to a different parent containing `DEC(0,m=2)`.
The target digest changed, but the factors matched. Warm transfer took `1`
proposal on all seeds versus `10--23` fresh proposals, and beat the prior
whole-candidate policy on every seed. Held-out accuracy was `1.0000` for all
targets; a reversal context learned `CDEC`, the original context retained
`CINC`, protected files and policy payloads reloaded exactly, shuffled feedback
was rejected, and replay/controller updates were zero. Evidence is archived
under
`session_records/recipe_factorized_context_proposal_credit_promoted_2026-08-12/`.

This promotes bounded factorized transfer and local reversal routing. It does
not yet establish reusable multi-step sub-sequence composition, unrestricted
memory growth, or general continual learning.

## Verifier-gated multi-step external composition

`ExternalRecipeCompositionMemory` is the next external CPU/files boundary.
It composes two existing immutable recipe files, checks that the candidate
program is exactly the stated ordered concatenation, and commits it only after
a stable verifier prefix passes. Each admitted composite retains immutable
left/right digest and order provenance. The controller and generic recipe
interpreter remain frozen; an optional composition policy receives only
opaque factor descriptors and aggregate scalar quality.

The four-seed audit uses mixed per-family moduli `(2, 8)` and an order-sensitive
conditional increment. Depth-two and depth-three compositions both reach
`1.0000` on held-out states, all sources retain `1.0000`, reversed programs
score `0.0000`, empty evidence cannot mutate memory, and shuffled verifier
feedback does not admit a file. Reloaded memory and policy checksums are exact;
replay and controller optimizer updates are zero.

This promotes bounded verifier-gated external composition, not learned
unrestricted program induction. The optional proposal policy's warm/fresh
sample-efficiency diagnostic is `[1.0, 3.0, 1.0, 1.0]` in proposal-count
ratios across seeds, so it remains a follow-up rather than a promoted
transfer claim. Evidence is archived under
`session_records/recipe_composition_growth_promoted_2026-08-12/`.

## Recursive external composition through depth four (2026-08-12)

The composition seam now carries generic recursive provenance and a
versioned structural descriptor: source depths and composite/atomic shape are
available to the replaceable outcome-only policy without exposing task names,
verifier rows, or semantic labels. Memory validation recursively reconstructs
every composite from earlier immutable files, rejecting forward references,
cycles, and rewritten provenance. Legacy factor-only policy payloads migrate
through a checksum-verified compatibility path.

The four-seed mixed-domain audit admitted four protected atomic files, then
grew verified depth-two, depth-three, and depth-four files with zero replay and
zero controller updates. Held-out accuracy was `1.0000` at every depth and
source retention stayed at `1.0000` on all seeds. Reversed programs failed
behaviorally, missing evidence was a no-op, shuffled feedback was rejected,
and memory/policy reloads were exact. The generic recursive gate accepts the
parent composite on either operand; this is necessary because the composition
mode describes instruction order, not parent-side identity.

The warm/fresh proposal ratios were `0.2917`, `1.2500`, `0.3200`, and
`0.3810`. Because one seed was slower than fresh, structural proposal transfer
remains diagnostic rather than promoted. The promoted claim is bounded
replay-free verifier-gated recursive external composition through depth four,
not arbitrary program induction, unrestricted memory growth, or general
continual learning. Evidence and accounting are archived in
`session_records/recipe_recursive_composition_growth_promoted_2026-08-12/`.

## Non-commuting recursive composition (2026-08-12)

The recursive audit now uses a dependency chain rather than mostly commuting
operations: each later file reads a slot changed by an earlier file. The
mixed-domain sources are `INC(0,m=2)`, `CINC(1|0,m=4)`, `CINC(2|1,m=8)`, and
`CDEC(0|2,m=2)`. Across four seeds, depth-two through depth-four held-out
accuracy and protected-source retention were all `1.0000`. The reversed
depth-four order scored `0.0625` on every seed, while provenance, reload,
missing-evidence, shuffled-feedback, and zero-replay gates passed.

Orientation-invariant shape/depth factors are now part of the versioned policy
ABI, with migration from the prior factor-only payloads. The warm/fresh
proposal ratios in this stronger chain were `0.8750`, `0.8421`, `3.0000`, and
`1.1111`; because the policy is not yet a reliable efficiency improvement, it
remains diagnostic. This promotes bounded replay-free recursive composition
with genuine order dependence, not arbitrary program induction or general
continual learning. Evidence is archived in
`session_records/recipe_noncommuting_recursive_composition_promoted_2026-08-12/`.

## Provenance-closed recursive compaction (2026-08-12)

`ExternalRecipeCompositionMemory.compact_verified()` adds the missing finite
capacity transaction for executable recipe files. It creates a copy-on-write
candidate containing the requested roots, their complete transitive
provenance closure, and all protected files. An independent behavior verifier
must accept the candidate before the caller adopts it; rejection leaves the
source memory byte-identical.

The four-seed audit populated ten files with a protected non-commuting
depth-four chain and three unreferenced decoys. Every run compacted to the
seven-file closure, removed all three decoys, retained the root and protected
sources at `1.0000`, reloaded exactly, and passed the rejected-no-op and
zero-replay controls. This promotes safe bounded storage compaction, not
learned eviction economics, semantic compression, unrestricted growth, or
general continual learning. Evidence is archived in
`session_records/recipe_recursive_compaction_promoted_2026-08-12/`.

## Learned recipe victim choice — rejected promotion (2026-08-12)

The generic `ExternalCapabilityEvictionPolicy` was connected to real
provenance-closed recipe compaction. Its input was limited to permutation-safe
structural telemetry: recursive depth, program length, protection, provenance
reference count, closure size, composite/root shape, and bank size. It learned
from one scalar compaction-verifier utility per fresh episode while the recipe
memory, verifier, interpreter, and controller remained frozen.

Across four seeds, depths two through four trained to `1.0000` transfer
accuracy on unseen depth-five files, versus a fresh mean of `0.6748`. The
candidate order was permuted, policy reload was exact, replay was zero, and
controller updates were zero. This is useful evidence that the external
storage seam can host learned maintenance state.

The promotion was rejected because the reward-shuffled null was not stable:
three seeds failed to transfer, but one also reached `1.0000` by drifting toward
a static candidate class. The aggregate null mean was `0.25`, but it does not
support a clean per-seed causal claim. The candidate-role distribution and null
control must be strengthened before learned eviction is promoted. Evidence is
archived at
`session_records/recipe_learned_eviction_rejected_2026-08-12/`.

## Context-conditioned learned recipe maintenance (2026-08-12)

The maintenance pressure test was strengthened rather than promoted on the
degenerate three-candidate result. The new rung uses two independent
recursive roots and two capacity-pressure regimes; the verifier-required root
changes between the two depth ranks. The replaceable external policy receives
only generic pressure context and permutation-safe structural telemetry, and
learns from one scalar compaction utility while the recipe memory, verifier,
interpreter, and controller remain frozen.

Training on depths 2–3 and transfer on unseen depths 3–4 reached `1.0000`
transfer accuracy on all four seeds. Fresh accuracy averaged `0.4922`, the
reward-shuffled null was exactly `0.5000` on every seed, corrupted features
were near chance, policy reload was exact, and the stable `0.90` threshold was
reached at update `64` (`55,296` verifier bits) on every seed. Replay and
controller optimizer updates were zero.

This promotes bounded context-conditioned learned external maintenance only.
It does not establish universal eviction economics, semantic compression,
unrestricted memory growth, or general continual learning. Evidence is
archived at
`session_records/recipe_learned_eviction_promoted_2026-08-12/`.

## Counterfactual learned recipe maintenance through four candidates (2026-08-12)

The sampled three-candidate rung exposed a credit-assignment bottleneck: final
accuracy was unstable across seeds even though the memory and telemetry seam
were correct. The maintenance learner now has an explicit counterfactual mode:
for each fresh lifetime it evaluates every candidate eviction through the
authoritative verifier and trains on the resulting scalar utility vector. This
does not expose verifier rows or controller state; it only spends additional
fresh verifier outcomes to reduce action-credit variance.

Across four seeds and four independent candidates, transfer accuracy was
`1.0000` on every seed and the stable `0.90` threshold was reached at update
`64` (`165,888` verifier bits). The reward-shuffled null was exactly the
four-way floor (`0.7500`), corrupted-feature accuracy averaged `0.7324`,
candidate order was permuted, reload was exact, and replay/controller updates
were zero. A scale-aware `0.20` causal margin is used because the chance floor
changes with candidate count.

This promotes bounded counterfactual utility learning for four-candidate
external maintenance. It does not establish universal eviction economics,
semantic compression, unrestricted memory growth, or general continual
learning. The verifier cost grows linearly with candidate count. Evidence is
archived at
`session_records/recipe_learned_eviction_four_counterfactual_promoted_2026-08-12/`.

## Repeated fixed-capacity recipe replacement (2026-08-12)

The next audit moves from isolated eviction lifetimes to an evolving external
bank. A frozen counterfactual policy repeatedly selects one active recursive
root for replacement; a copy-on-write verifier must preserve every other
mastered root, all protected source files, and the incoming root. Eight
replacements are performed at fixed capacity, then the same stream is replayed
against reversed physical source order.

Across four seeds, all `8/8` forward and reversed replacements committed,
protected sources remained present, checksummed reloads retained active roots,
and rejected transactions left the source memory unchanged. The bank stayed at
`28` files. The trained policy reached stable `>=0.90` victim selection at
updates `128`, `256`, `128`, and `128`, while a fresh policy and a
reward-shuffled policy each accepted only `2/8` stream selections. The
counterfactual objective is explicitly `evict`; the underlying verifier still
checks retention and controls adoption.

Accounting is `2,585,600` unique verifier bits and `1,024` optimizer updates
per seed, with `1,056` unique logical lifetimes, zero replay, and zero
controller updates. This promotes bounded repeated verifier-gated external
maintenance and fixed-capacity replacement. It does not establish unrestricted
memory growth, semantic compression, arbitrary new computation, or general
continual learning. Evidence is archived at
`session_records/recipe_repeated_maintenance_promoted_2026-08-12/`.

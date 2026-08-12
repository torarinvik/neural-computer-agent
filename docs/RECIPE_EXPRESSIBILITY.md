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

## Stochastic multi-edit control-flow frontier

The typed external CPU boundary now has a persistent stochastic frontier with
generic replace, insert, delete, and swap edits. It retains a protected root
plus scalar-quality-qualified provisional files, retries other parents when a
local neighborhood is exhausted, and stores only aggregate outcome credit and
content-addressed program files. A fresh frontier termination is reported as
`frontier_exhausted` or `budget_exhausted`; neither is silently promoted to
`inexpressible`.

Across four seeds and forward/reversed verifier order, a useful clear-loop
root acquired a two-edit transfer program with `1.0000` held-out accuracy and
`1.0000` source retention. The same-ABI fresh root exhausted its qualified
frontier after `50` evaluations without reaching the target. Missing evidence,
shuffled feedback, corruption, frontier reload, memory reload, zero replay,
and zero optimizer updates all passed. The fresh arm did not reach a common
threshold, so no warm/fresh transfer ratio is claimed; this is a structural
acquisition promotion, not a sample-efficiency promotion.

This moves beyond one-edit scaffold adaptation, but remains a bounded
stochastic frontier over a small counter-machine file. It does not establish
efficient arbitrary program synthesis, unbounded execution, unrestricted
memory growth, or general continual learning. Evidence is archived in
`session_records/recipe_control_flow_frontier_growth_promoted_2026-08-12/`.

## Canonical typed control-flow intention bridge

`ControlFlowProgramAmodalRuntime` now connects the typed counter-machine file
to the canonical `AmodalControllerRuntime`. The independently versioned
`ControlFlowIntentionAdapter` is the only component that translates between
opaque learned intentions and integer counters. The controller itself never
sees instruction pointers, counters, program slots, or control-flow opcodes.
After bounded execution, the adapter returns an opaque `IntentEvent`, which is
validated and fanned out through the existing decoder bus.

The bridge has separate checksummed controller/external state, memory-backed
file support, explicit step/counter limits, and no controller parameter
updates. Regression coverage verifies frozen-core execution, batch
permutation, exact reload, checksum rejection, and ABI-width rejection. This
is an interface integration result, not a learned program-routing or general
continual-learning promotion. The next pressure is learned codec adaptation
and multi-file composition beyond this two-file routing boundary.

## Multi-file typed control-flow routing — narrow outcome-only promotion

The canonical bridge now accepts the existing generic
`ExternalOutcomeProgramRouter` over opaque intention features and maintains a
separate counter state for each external file. Route choices, exact
propensities, per-file execution digests, and delayed scalar feedback remain
outside the controller; the decoder still receives only the resulting opaque
intention. Boundary tests cover mixed batch routing, isolated file state,
checksummed routed-state reload, and route-state corruption rejection.

The canonical bridge was then exercised with the generic
`ExternalOutcomeProgramRouter`: two protected files retained separate counter
state, route propensities and per-file digests were emitted, and checksummed
pause/resume and corruption rejection passed. The controller, adapter, and
external files were frozen; the router received one delayed scalar outcome per
fresh episode through the optional route-only feedback channel, with zero
replay and zero controller optimizer updates. The controller therefore saw the
same quiet feedback distribution during training and evaluation.

Across four seeds and both forward/reversed physical-file orders, all eight
verifier arms reached `1.0000` held-out accuracy. Stable `>=0.80` training
prefixes were measured at `119`, `1000`, `1`, `999`, `1`, `1`, `1`, and `1`
fresh verifier bits. The paired reward-shuffled order-permutation null was
exactly `0.5000` for every seed; individual symmetric arms sometimes drifted to
`0.0000` or `1.0000`, so the paired permutation is an explicit promotion gate
rather than an omitted control.

This promotes bounded outcome-only routing among two generic external files
with frozen controller and isolated file state. It does not promote arbitrary
new computation, unrestricted memory growth, or general continual learning.
The earlier shared-feedback audit remains archived as a rejected diagnostic at
`session_records/control_flow_runtime_routing_rejected_2026-08-12/`; the
promotion report is archived at
`session_records/control_flow_runtime_routing_promoted_2026-08-12/`.

## Full-information four-file route credit — bounded promotion

The next route-bank rung adds a replaceable
`ExternalControllerTrajectoryQueryAdapter` and four protected generic files.
For each fresh lifetime, the verifier evaluates every active file and sends
the external router a full outcome vector. This spends four fresh verifier
outcomes per lifetime to remove sampled-action credit variance; the controller
still receives quiet feedback and remains frozen.

Across four seeds and both forward/reversed physical-file orders, all eight
verifier arms reached `1.0000` held-out accuracy on amplitude-2 events, from a
`0.2500` fresh baseline. The paired independent-random-outcome nulls were
`0.1250`, `0.1250`, `0.2500`, and `0.2500`, within the predeclared `+/-0.15`
band around the four-file chance floor. Protected files, frozen controller,
zero replay, zero controller optimizer updates, exact reload, corruption
rejection, and missing-evidence no-op gates all passed.

This promotes bounded full-information outcome-only routing among four
generic external files with an opaque trajectory address. It does not promote
learned codec adaptation, arbitrary new computation, unrestricted memory
growth, or general continual learning. Evidence is archived in
`session_records/control_flow_runtime_four_file_counterfactual_promoted_2026-08-12/`.

## Sequential context-conditioned external-file growth — bounded promotion

The canonical control-flow runtime now consumes a replaceable
`PersistentOpaqueContextRouteEvidence` table directly at its opaque route
query boundary. Four protected generic files were acquired one context at a
time using only the scalar verifier outcome for the externally selected file;
old contexts received no replay while later contexts were learned. A final
single-context reversal changed the correct file and exercised external
recovery without mutating the other context rows.

Across three seeds and both forward/reversed physical-file orders, all six
verifier arms reached `1.0000` held-out accuracy on all four contexts after
growth and after reversal. Fresh banks scored `0.2500`; reward-shuffled nulls
also scored `0.2500`. The controller stayed byte-identical, all files stayed
protected, route and runtime reloads were exact, checksum corruption was
rejected, and replay/controller optimizer updates were zero. The short rung
used `160` scalar verifier bits per arm, with `32` fresh lifetimes per context
and reversal.

The first run exposed and fixed an important reversal flaw: lifetime-average
promotion permanently poisoned a candidate that had failed before a
nonstationary reversal. Persisted recovery streaks now allow a fresh stable
success run to promote that candidate while retaining unrelated evidence. This
promotes bounded sequential context-conditioned external-memory growth and
reversal recovery only; it does not establish unrestricted memory growth,
content search, arbitrary new computation, or general continual learning.
Evidence is archived in
`session_records/control_flow_runtime_context_conditioned_growth_promoted_2026-08-12/`.

## Gated related-context route transfer — bounded promotion

The external route table now has an opt-in `generalization_tolerance`. When a
new learned trajectory query lies within that distance of a protected context
with a stable preferred file, it may borrow that preference as a cold-start
prior. The new query is not aliased to the old row: its first observed scalar
outcome creates independent evidence, and a local reversal can override the
prior without changing the source context.

Across three seeds and both physical-file orders, an unseen related query
transferred at `1.0000` versus `0.0000` for a matched fresh table. A distant
query stayed on append-order fallback. After the related context's correct file
reversed, it relearned at `1.0000` while the original source remained at
`1.0000`. Reward-shuffled arms transferred at `0.0000`; route persistence,
checksum rejection, protected-file retention, frozen controller, and zero
replay/controller updates passed. Each verifier arm used `64` unique scalar
verifier bits and `64` logical lifetimes.

This promotes bounded metric-neighborhood prior reuse and isolated local
reversal. It does not establish semantic relatedness, robust representation
migration, content search, unrestricted memory growth, arbitrary new
computation, or general continual learning. Evidence is archived in
`session_records/control_flow_runtime_related_context_transfer_promoted_2026-08-12/`.

## Route-query representation migration guard

The route evidence payload now records a versioned `query_space_id`, and
`ExternalControllerTrajectoryQueryAdapter` declares the same identity. The
canonical control-flow runtime rejects a route table paired with an
incompatible query space before it can select or execute a file. This closes a
silent-failure mode in which equal-width but differently trained projections
could reinterpret every persisted address. Legacy payloads without the field
remain readable as `opaque-route-query-v1`; changing the learned query ABI
requires an explicit version bump and a fresh or migrated evidence table.

This is an interface-integrity safeguard, not a learning-capability claim.

## Outcome-only structural acquisition through the canonical runtime

The structural control-flow frontier is now exercised end to end through the
production amodal boundary. It first acquires a generic external
counter-machine file from scalar verifier outcomes, then admits that file
beside protected source/decoy files. A frozen `AmodalCognitiveController`
emits opaque intentions; `ControlFlowProgramAmodalRuntime` routes and executes
the selected file through checksummed `PersistentOpaqueContextRouteEvidence`,
and returns only an opaque intention to the output bus.

Across three seeds and both physical-file orders, acquired-file held-out
mastery, canonical route selection, canonical execution, and source retention
were all `1.0000`; a matched fresh acquired-file control was `0.0000`, and
reward-shuffled route mastery was `0.0000`. The controller and files remained
byte-stable, with zero replay and zero controller optimizer updates. The
opaque counter codec derives its bounded input from the complete opaque
intention, preventing a single-coordinate zero from making source and target
files observationally identical.

This promotes bounded outcome-only structural acquisition followed by
canonical frozen-controller execution and route learning. It does not
establish arbitrary program induction, unrestricted memory growth, or general
continual learning. Evidence is archived in
`session_records/control_flow_runtime_acquired_program_promoted_2026-08-12/`.

## Reusable external composition through the canonical runtime

The typed control-flow ABI now materializes a sequential composition from
existing external files. It relocates internal jump targets, maps each
component's terminal halt to the next file, rejects incompatible counter
widths and ambiguous internal halts, and admits the resulting ordinary file
through the same scalar stable-prefix verifier.

The canonical audit first acquired a transfer loop, composed it with a second
external increment file, and routed the resulting artifact through the frozen
amodal runtime. Across three seeds and both physical-file orders, component
and composed held-out mastery, composed route/execution, and source retention
were all `1.0000`; fresh and reward-shuffled controls were `0.0000`. The
controller and external files stayed unchanged, with zero replay and zero
controller optimizer updates. Evidence is archived in
`session_records/control_flow_runtime_composed_program_promoted_2026-08-12/`.

This promotes bounded reusable external composition, not arbitrary program
induction, unrestricted memory growth, or general continual learning.

## Outcome-only composition search through the canonical runtime (2026-08-12)

The remaining manual assumption in the preceding audit was the ordered factor
list: the caller still supplied which files to compose.  The new
`ControlFlowCompositionSearch` enumerates opaque file-slot sequences,
materializes each candidate through the generic control-flow ABI, and admits
only a candidate whose scalar verifier prefix remains above threshold.  Its
restartable state is bound to the checksummed file-memory digest and stores
only scoped candidate identities and aggregate quality; individual verifier
rows never enter durable state.  Changing the underlying file memory
invalidates the search state.

Across three seeds, forward/reversed file order, and verifier/reward-shuffled
route arms, the canonical audit admitted a searched composition, reached
`1.0000` component/composed held-out mastery, route/execution, and source
retention in every verifier arm, and kept the controller frozen with zero
replay and zero controller optimizer updates.  The search evaluated seven
opaque two-file candidates before admission in this bounded neighborhood.  In
one valid run it selected a behaviorally equivalent `(fresh, acquired)` pair
rather than the provenance expected by the old hand-written factor list; the
held-out verifier correctly accepts behavior, not semantic provenance.  This
is a stronger boundary than manual composition but still bounded search over
existing files, not general program induction, unrestricted memory growth, or
general continual learning. Evidence is archived in
`session_records/control_flow_runtime_composition_search_promoted_2026-08-12/`.

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

## Synchronized external program-file growth (2026-08-12)

The canonical runtime now exposes one copy-on-write admission transaction for
an expanding control-flow file bank. `admit_program_verified()` stages the new
checksummed executable file together with its route capacity/state, opaque
context evidence, and per-file counter state. Nothing is committed until the
scalar verifier prefix is stable. A rejected candidate returns the original
runtime state and leaves every external object unchanged.

The route-capacity branch grows only after an exact structural retention check
of every pre-existing router column. The evidence branch appends one opaque
slot to every existing context row. In both cases the frozen controller and
the existing executable files remain byte-identical. The API therefore closes
the synchronization seam between external memory growth and runtime use; it
does not make program induction autonomous.

The three-seed audit passed evidence-backed growth, router-capacity growth,
target execution, source retention, exact memory reload, rejected-candidate
rollback, synchronized counters, and frozen-controller controls. It charged
`18` unique verifier bits per seed, with zero optimizer updates and zero
replay. Evidence is archived at
`session_records/control_flow_runtime_program_growth_promoted_2026-08-12/`.

This promotes a narrow external-memory lifecycle contract. It remains bounded
growth of verifier-admitted files, not unrestricted memory growth, arbitrary
program induction, or general continual learning.

## Runtime-owned route credit and autonomous reachability (2026-08-12)

The route lifecycle now persists the previous opaque query and selected file
in `ControlFlowRuntimeState` v2. The next step can supply an explicit scalar
route outcome without repeating a slot ID; the runtime stages that credit into
the external context evidence before selecting the next file. With nonzero
exploration, a newly admitted file is sampled by an external epsilon-greedy
distribution and its exact propensity is exposed for accounting.

Across three seeds, two contexts were interleaved after a new file was
admitted. Both context bindings were learned without overrides, one binding
was reversed without replaying the other, and the unreversed binding stayed
correct. State/evidence reload, checksum, protected-file, frozen-controller,
exact-propensity, zero-replay, and shuffled-feedback controls passed. The
positive and shuffled arms charged `400` verifier lifetimes each per seed.
Evidence is archived at
`session_records/control_flow_runtime_autonomous_route_growth_promoted_2026-08-12/`.

This promotes bounded route reachability and external credit assignment. It
does not establish unrestricted memory growth, arbitrary program induction, or
general continual learning.

## Balanced open-world external-file discovery (2026-08-12)

The first eight-context audit exposed a real acquisition problem: novelty
weighting alone left an unseen context with a uniform lottery over eight files,
so the newest target could receive no trial. The external route policy now has
a balanced mode that samples among the least-attempted files until a context
has a protected winner. This keeps candidate coverage broad as the file bank
grows while preserving exact propensities and exploit-plus-novelty behavior
after mastery.

Across three seeds, eight files were admitted incrementally and eight unseen
contexts were learned in an interleaved stream. All eight context-to-file
bindings were correct, including the newest file. One binding reversed while
the other seven remained correct; context keys remained distinct; protected
files, reload, corruption, frozen-controller, zero-replay, and shuffled
feedback controls passed. The audit charged `462` verifier bits per arm and
seed, or `2,772` across the six positive and shuffled arms, with zero optimizer
updates. Evidence is archived at
`session_records/control_flow_runtime_open_world_route_growth_promoted_2026-08-12/`.

This promotes bounded open-world route discovery and acquisition coverage. It
does not establish unbounded memory growth, arbitrary program induction, or
general continual learning.

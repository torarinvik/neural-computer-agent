# `neural_computer`

This is the canonical production package. It owns only versioned neural-IR
contracts and modality-independent runtime composition:

```text
N encoders -> event-token window -> one controller/memory
           -> intention bus -> M decoders
```

Raw modality frontends and protocol backends are independently supplied by the
caller. Historical controller implementations are archived under
`experiments/archive/` and must not be imported by production code.

The package includes a concrete keyboard boundary for interactive agents.
`KeypressEncoder` maps an externally logged key index to a learned opaque
feedback vector before it reaches the controller. `KeypressDecoder` maps an
`IntentEvent` to logits over the external key index space and reports the
exact sampled propensity for outcome-only credit. Neither class adds a
keypress-specific reasoning branch: replacing the keyboard with another
decoder leaves the controller interface unchanged.

The package also exposes trainer-only protected-plasticity utilities. They
accumulate verified rehearsal gradients and remove only target-update
components that oppose the protected direction; they never add task labels,
semantic fields, or extra reasoning branches to the deployed controller. The
mechanism is intended for zero-impact growth adapters and continual-learning
trainers, where old-capability retention is a hard promotion gate.

The trainer-only counterfactual credit utilities provide a reusable outcome
boundary for those learners. Common-random hidden worlds can produce paired
scalar verifier outcomes, then `paired_counterfactual_policy_loss` assigns
credit to a binary opaque decision while
`paired_counterfactual_ranking_loss` gives bounded preference credit to two
attempted memory/artifact rows. Pairing metadata and interventions stay in the
trainer; the deployed controller sees neither labels nor counterfactual arms.

`ExternalGrowthPrior` is the reusable initialization boundary for external
growth modules. It averages validated adapter state, loads it copy-on-write
into a fresh compatible adapter, can reset a capability-specific output head,
and rejects namespace, shape, dtype, or non-finite drift. Existing capability
artifacts and the shared controller are never mutated by prior updates. This
is safe external transfer state, not controller plasticity or a claim of
sample-efficient general continual learning.

`ExecutableArtifactMemory` is the canonical hot/cold store for opaque learned
growth artifacts. It addresses artifacts with controller-produced learned
keys, stores only tensor payloads plus integrity hashes, and supports atomic
save/reload, verified single- or top-k promotion, hot eviction, and
behavioral-test compaction. An alias may additionally carry an opaque view
identifier; promotion returns that identifier with the verified handle, while
the storage backend remains unaware of its meaning. Top-k promotion exposes
only verified opaque candidates; it does not interpret their tensor payloads or
decide whether a caller should compose them. `view_candidates()` and
`promote_view()` expose the same boundary for a replaceable learned router
when a physical row contains multiple executable views.
The memory backend does not execute or interpret an artifact: a caller loads
the verified payload into a generic growth state while the shared controller
remains frozen. This keeps persistent learned programs independently
replaceable from the controller and prevents task-specific logic from entering
the runtime boundary.

`ExternalCapabilityLifecycle` is the canonical coordinator for the mutable
artifact boundary. It composes opaque admission planning, protected-row
eviction, capacity growth, and verified consolidation into one versioned
transaction surface; failed plans and failed verifiers leave the source store
unchanged. It does not execute artifacts, choose semantic routes, or update
controller weights. The promoted two-step growth audit uses this coordinator
for both additions and passes frozen-core, retention, permutation, reload,
corruption, shuffled-outcome, and zero-replay controls.

`ConfidenceAwareCapabilityStaging` is the fail-closed admission boundary for
new growth. It keeps an opaque candidate artifact outside the executable bank
until deterministic verifier outcomes clear a stable-prefix mastery gate, then
transfers the accumulated scalar evidence into the destination retention ledger
without replaying old episodes. A candidate that has not earned stable mastery
cannot evict or dilute an existing protected row. The queue is in-process by
default, and an optional staging directory adds atomic artifact snapshots,
checksums, and restart recovery. Admission itself remains a separate
executable-memory transaction, so the API does not overclaim a multi-process
distributed commit protocol.

`select_capability_candidate` applies the same fail-closed idea when multiple
replaceable learners compete for a new task. It selects only a unique
stable-prefix winner from fresh held-out curves; unstable candidates and ties
are rejected. This lets inherited composition state compete against a fresh
candidate without allowing a lucky final score or a prior to silently win.
`ExternalCapabilityLifecycle.admit_selected_candidate` makes that decision
atomic with admission: a rejected selection leaves the executable bank
unchanged, while only the selected opaque artifact is passed through the
normal capacity and protection policy.
The executable compaction audit now uses the same coordinator for all three
rewrites and defaults its separate opaque route-acquisition arm to 2,048 fresh
paired-outcome updates per arm after a 512-update control failed its
permutation gate.

`freeze_core` and `load_growth_artifact` enforce the complementary rehydration
boundary: only explicitly prefixed growth state may be loaded into a module,
and the loader hashes all remaining state before and after the copy. A memory
artifact therefore cannot silently overwrite the shared processor while it is
being updated or restored.

`OpaqueAddressRouter` is an optional memory-side resolver for cases where a
controller query cannot directly identify a physical row. It is
permutation-equivariant over variable candidate rows and trains only from an
attempted-row scalar outcome. `FactorizedOpaqueAddressRouter` is the preferred
learned-address variant: it independently embeds opaque queries and keys into
a shared latent space before matching them, which makes outcome-only address
discovery substantially more sample-efficient without assigning meaning to
key coordinates. `ExecutableArtifactMemory.address_rows()` and
`promote_index()` keep row selection separate from artifact verification and
growth-state execution. These are replaceable memory policies, not
modality-specific branches in the controller.

`PersistentOpaqueStateStore` is the durable state boundary for those learned
memory policies. It atomically snapshots a route or utility policy's tensor
state together with a versioned JSON configuration and SHA-256 state digest;
`load_module()` validates the contract before mutating the replacement module.
This keeps learned address weights independent from both the frozen controller
and executable artifact files, so a restart restores the policy that makes the
files usable rather than only restoring the files themselves. It stores no
task labels, modality formats, protocol IDs, or semantic coordinate meanings.

`PersistentOpaqueRouteEvidence` is the small persistent policy for append-only
capability banks. It stores only opaque slot attempts and scalar verifier
outcomes, and promotes a slot to preferred status only after a stable-prefix
candidate gate. The route ledger is external mutable state: it can grow,
serialize, and be replaced without changing controller weights. Its current
Brain Workshop result promotes bounded three-slot growth for 16-step
lifetimes, while cue-free short-lifetime task inference remains explicitly
unqualified.

`PersistentOpaqueContextRouteEvidence` extends that policy with a learned
event-key index. It keeps one independent opaque evidence ledger per matched
context, uses append-order fallback for unknown keys, and separates route
reads from explicit calibration writes. The canonical Brain Workshop audit
promotes three cue-conditioned short-lifetime slots without changing the
controller; cue-absent task inference remains unqualified.

`ExternalSkillFragmentBank` is the compositional growth boundary. Its rows are
reusable opaque coefficient sequences over one shared operator basis, not
task-sized policy modules. A learned event/intention query routes to fragment
rows, and `ExternalCapabilityRegisterMachine.execute_fragment_composition_trace()`
executes the resulting variable-length chain after removing transport padding
while preserving an opaque post-instruction state trace. The external
`ExternalSkillFragmentCombiner` can consume that trace before an output decoder
without unfreezing or resizing the controller.
Appending a fragment grows only external state; it does not resize the
controller, interpreter, or decoder. If the current shared basis is saturated,
`grow_basis()` appends new instruction directions and zero-pads old
coefficients, while `freeze_basis_prefix()` protects mastered directions from
later candidate updates. The bank supports outcome-only route refinement, row
permutation audits, protected rows, atomic disk persistence, and checksum
recovery. Fragment codes are normalized at materialization so small learned
coefficient/basis products cannot collapse the external instruction signal.
This is the structural foundation for compounding reuse; positive transfer and
arbitrary program induction still require fresh verifier-gated experiments.

`ExternalSkillFragmentSerialCombiner` is the stricter execution-state variant.
It keeps a replaceable opaque state vector and applies one learned state
transition at each fragment boundary. `step_sharing="position"` gives each
position its own protected external slot; `step_sharing="shared"` reuses one
transition across arbitrary lengths. Slots append with zero impact, mastered
prefixes can be frozen, and the whole state persists through a versioned,
checksummed payload. This is the intended CPU-plus-files execution seam, but
the source-mastered ordered-composition audit still failed to learn the
execution law from final outcomes alone; the class is therefore infrastructure,
not a promoted general-learning result.

`forward_prefixes()` returns the serial external state after each opaque
fragment boundary for verifier-gated trainers. It is intentionally not part of
the controller ABI and carries no target, operation, route, or verifier
metadata. Directly training a shared decoder on those snapshots failed to
produce ordered transfer; retain the API for causal interventions, not as a
claim that intermediate decodability solves credit assignment.

`forward_with_gates()` and `forward_leave_one_out()` expose the corresponding
trainer-only causal seam: an opaque external transition-use policy can gate a
serial state, and each transition can be omitted for a paired final-outcome
intervention. The deployed controller receives neither the intervention nor
its verifier result. The first full audit was rejected for insufficient
held-out order transfer, so this remains an attribution primitive rather than
a promoted learning capability.

The multi-fragment closure pressure test in
`experiments/external_skill_fragment_composition_amodal/train_multi.py` adds a
crucial lifecycle rule: acquire and stably master one primitive, protect it,
then acquire the next. Primitive acquisition and composition acquisition use
separate objectives and separate output adapters. This prevents a longer
program from corrupting the reusable primitive it is supposed to compose. The
four-fragment audit promotes bounded sequential acquisition and held-out
composition transfer; arbitrary program induction and open-ended growth remain
unqualified.

`ExternalGoalFragmentStager` is the corresponding acquisition boundary for
factual destinations. It stages an opaque learned-state target and updates
only scalar sufficient statistics from fresh eligible verifier outcomes. A
stable-prefix gate then passes the candidate through
`ExternalGoalFragmentMemory`'s copy-on-write retention probe. The stager never
stores verifier rows or replayable trajectories and cannot mutate the
controller. This supports learn-while-frozen destination growth while leaving
general goal discovery and unrestricted continual learning as open empirical
questions. Use `goal_fragment_candidate_from_controller_output()` when the
candidate comes from the live learned state: it forces the external state
adapter to establish planner-space width before the stager accepts evidence.

`PersistentOpaqueContextRouteEvidence` now closes the first execution loop for
those fragments. The policy-free runtime can synchronize append-only goal slots,
learn a context-conditioned preferred slot from opaque scalar outcomes, and
select separate fragments for separate batch rows through
`ExternalGoalFragmentMemory.propose_per_batch()`. Unknown contexts use
append-order fallback, and repeated low outcomes can demote a stale protected
route without deleting the underlying fragment. Route state is external,
versioned, and serializable; the controller remains frozen and receives no
fragment address. This is a verified routing boundary, not yet general goal
discovery or arbitrary new computation.

`PolicyFreeAmodalRuntime.transition_observation()` is the matching world-model
seam. Given consecutive policy-free outputs, it creates an opaque
`ExternalTransitionObservation` containing only learned planner state,
decoder-facing intention, next learned state, and optional generic
confidence. External transition-model banks can consume that row without
letting protocol actions or verifier answers enter the controller. Recursive
held-out model rollout and fresh-learner transfer remain required before
claiming general continual learning. `learn_transition_once()` enforces the
replay-free affine/random-feature bank path and refuses a replay-dependent
neural slot.

`ExternalBoundTransitionModel` is the bind-once execution view for a
contextual factual model. It captures one opaque external context before a
multi-step rollout, keeps that binding stable while the planner iterates, and
preserves exact content-addressed hit evidence. Passing `require_known=True`
to `ExternalModelBasedPlanner` or `PolicyFreeAmodalRuntime.step_events()`
then rejects missing transition rows before they can win beam search. A
continuous learned model can still use the compatibility path, but it cannot
claim exact read coverage without a `predict_with_hit` implementation. This
is an unknown-rejection and execution-integrity boundary, not evidence that
the model has learned arbitrary dynamics.

The same audit also supports a two-family nonstationary rung: n-back-2 source
experience and n-back-3 target experience occupy separate opaque external
contexts. The source model remains byte-stable while the target model beats a
matched fresh bank with zero replay. This is isolated factual-model retention,
not yet end-task acquisition or arbitrary goal discovery.

The online discovery rung then starts with only the source transition slot.
Target rendered transitions are staged under a newly inferred opaque context,
learned once by the replay-free affine bank, and committed only after held-out,
recursive, source-retention, and matched fresh-candidate gates. A passing seed routes later target
lifetimes back to the new slot while the source remains byte-stable. This
boundary passes on one seed but is not promoted yet because neighboring seeds
are rejected by the candidate gate; the next work is robust discovery followed
by goal-conditioned end-task acquisition.

The rendered audit
`experiments/brainworkshop_canonical/replay_free_transition_acquisition.py`
now validates this path against fresh `NBackVerifier` lifetimes. Its default
rung lowers recursive held-out rollout error from `0.06614` in a matched fresh
bank to `0.02737` after 18 one-pass external updates, with zero replay and an
unchanged controller. This qualifies frozen-core factual-model acquisition;
it is not yet evidence of new-task goal discovery or general continual
learning.

The canonical runner exposes the route table through a versioned
`route_state_payload()` / `load_route_state_payload()` boundary. Reloading it
validates slot count and context width and does not load or mutate controller
weights.

The canonical Brain Workshop route payload is now `brainworkshop-route-state.v2`:
it also records the learned-event encoder configuration and a digest of that
encoder's state. A route table paired with an incompatible event
representation is rejected explicitly instead of silently treating every cue
as unseen. This keeps route memory external and replaceable while making
representation migration a separately testable ABI operation.

Route evidence also has explicit reversal patience: repeated low outcomes can
retire a stale context-to-slot preference while retaining the underlying
capability row. `observe_batch()` reduces repeated context/slot attempts to
one scalar per rollout batch, so patience is not coupled to the number of
eligible trials in a lifetime. The Brain Workshop failure-only demotion audit
is promoted across three seeds: fresh changed-task outcomes demote a protected
route without same-cue calibration, an already learned replacement becomes
preferred, and the old capability remains independently retained. A separate
same-cue replacement audit is also promoted. These are bounded
failure-driven external-memory results, not general continual learning.

The canonical control-flow runtime now consumes the same route evidence inside
its opaque selection seam. A three-seed, forward/reversed-file audit learned
four context-to-file bindings sequentially from selected-file scalar outcomes,
retained every earlier context without replay, and recovered one reversed
binding at `1.0000` held-out accuracy. The controller stayed byte-identical;
fresh and reward-shuffled controls were `0.2500`. Route memory is checksummed
and reloadable independently from runtime state. This remains bounded
context-conditioned external memory, not unrestricted growth or general
continual learning.

The audit caught a reversal failure in lifetime-average promotion: a candidate
with earlier failures could never recover. `PersistentRouteEvidence` therefore
persists a recovery streak so a fresh stable success run can promote a
previously bad slot without clearing other context rows. Evidence is archived
under
`session_records/control_flow_runtime_context_conditioned_growth_promoted_2026-08-12/`.

`PersistentOpaqueContextRouteEvidence` also exposes an opt-in gated
`generalization_tolerance`. A protected preferred route can seed a nearby
unseen trajectory query, while the new query remains an independent row and
can later reverse locally. The canonical three-seed audit reached `1.0000`
related transfer and `1.0000` local reversal with `1.0000` source retention in
both file orders; fresh related lookup was `0.0000`, distant lookup stayed on
fallback, and reward-shuffled transfer was `0.0000`. This is bounded
metric-neighborhood external-memory reuse, not semantic understanding or
general continual learning. Evidence is archived under
`session_records/control_flow_runtime_related_context_transfer_promoted_2026-08-12/`.

Route evidence and its replaceable trajectory-query adapter now share a
versioned `query_space_id`. The canonical runtime rejects mismatched IDs
before executing an external file, preventing silent address corruption after
an incompatible projection change. Pre-identity payloads migrate to the
explicit `opaque-route-query-v1` default; new query representations must bump
the ID and intentionally migrate or reset their external evidence. This is a
durability/ABI safeguard, not a capability claim.

### Outcome-only acquisition reaches the canonical runtime

The control-flow frontier now has a promoted end-to-end rung: scalar
verifier-only structural acquisition produces a generic external file, the
file is admitted beside protected source and decoys, and a frozen amodal
controller routes opaque intentions through `ControlFlowProgramAmodalRuntime`.
The runtime executes the selected external file and sends only the resulting
opaque intention to the decoder bus.

Across three seeds and both physical-file orders, acquisition, route
selection, execution, and source retention were `1.0000` in every verifier
arm. Fresh acquired-file and reward-shuffled controls were `0.0000`; the
controller/files stayed unchanged and replay/controller updates were zero.
This is bounded outcome-only structural acquisition plus canonical execution,
not arbitrary program induction or general continual learning. Evidence is
archived under
`session_records/control_flow_runtime_acquired_program_promoted_2026-08-12/`.

`ControlFlowCompositionSearch` removes the last manual factor-order assumption
from the canonical composition rung. It enumerates opaque ordered file-slot
sequences, exposes only scalar verifier outcomes to its admission evaluator,
and persists a memory-digest-bound state containing candidate identities and
aggregate quality rather than verifier rows. A changed file memory invalidates
the state before another proposal can be made. The promoted bounded audit
searched seven two-file candidates before admitting a held-out-mastered
composition across forward/reversed file order and verifier controls. The
accepted sequence may be behaviorally equivalent to another provenance order;
that is expected because the verifier establishes reusable behavior, not
hand-labeled factor identity. This remains bounded external composition, not
general program induction or unrestricted continual learning.

### Reusable external composition reaches the canonical runtime

The typed control-flow ABI can now compose existing external files by
relocating jumps and safely transferring terminal halts. The composed file is
admitted through scalar verifier evidence and remains outside the controller.
The canonical three-seed audit reached `1.0000` component/composed mastery,
route, execution, and source retention in both file orders; fresh and
reward-shuffled controls were `0.0000`, with zero replay and controller
updates. This is bounded reusable external computation, not arbitrary program
induction or general continual learning. Evidence is archived under
`session_records/control_flow_runtime_composed_program_promoted_2026-08-12/`.

The delayed score-function route credit path is batch-safe: each trajectory's
feature gradient is expanded across the action axis before updating its
independent eligibility tensor. This keeps batched route learning equivalent
to independent single-trajectory updates while preserving exact propensities.
Content-addressed context evidence remains the conservative choice when
unknown keys must fall back rather than generalize to a newly appended file.

`AdaptiveOnlineEpisodicRelationReader` is the generic capability blueprint for
the next growth rung. It scores each present event/action/outcome row before
mixing relation contexts, so one fixed external window can learn different
relation horizons without putting a task horizon in the capability
constructor. The canonical Brain Workshop audit provisions two slots with the
same capacity-five adaptive reader, learns n-back-3 and n-back-4 from fresh
outcomes, retains n-back-2, and passes cue, shuffle, reload, frozen-core, and
zero-replay controls across three seeds. This is bounded generic capability
growth; it still requires an observable cue and candidate calibration and is
not arbitrary program induction or general continual learning.

The same route boundary now supports automatic discovery: a newly acquired
generic slot can become preferred for an unseen rendered cue from ordinary
failure-gated fallback outcomes, without a forced candidate-calibration write
for that cue. The replicated Brain Workshop audit promotes this bounded
outcome-driven route promotion while preserving old capability retention and
route-state reload. It remains cue-conditioned external memory, not arbitrary
new computation or general continual learning.

`compose_growth_artifacts()` is the caller-owned execution-side merge for
verified payloads. It remaps artifacts into disjoint growth namespaces,
rejects collisions, and returns detached tensors for the generic loader.
`select_growth_artifact_view()` projects one opaque namespace after memory-side
view routing, so independently learned procedures do not execute
simultaneously merely because they share one compacted row. The working-memory
audit promotes this routed logical compaction across two seeds; naive
unrouted composition is explicitly rejected. This is a storage and execution
contract, not a claim of arbitrary procedure induction or general continual
learning.

`ExternalControllerTrajectoryQueryAdapter` is an optional memory-side address
adapter. Its compatibility mode augments the final opaque controller state
with masked mean/max statistics over the learned event-token window. Its
opt-in `recency_weighted_and_latest_v1` mode instead preserves causal order
through a recency-weighted summary and the latest retained token. Both allow a
growing router to retain more trajectory identity without changing the planner
state or adding a modality branch. `ExternalOutcomeIntentionRouter` also
supplies a bounded
exploration floor for unqualified cells so appended memory receives evidence
before route logits can suppress it.

The rejected six-regime audit is intentionally preserved as a design control:
copying an external intention-generator policy into a contradictory regime can
produce negative transfer. Long-term growth must therefore prefer factual
residual/delta candidates or fresh challengers and held-out copy-on-write
selection over blind policy cloning.

Retention is split into two authorities: noisy online outcomes may adapt
unprotected cells, while `ExternalOutcomeIntentionRouter.verify_and_protect`
is a held-out copy-on-write gate for freezing them. The verifier transaction
records a stable-prefix receipt and changes only protection/qualification
metadata; it never silently trains content or routing.

The replicated six-regime follow-up now passes bounded stable-prefix retention
and route-preservation gates with held-out qualification and verified prototype
addressing. Evidence is in
`session_records/policy_free_intention_prefix_growth_promoted_2026-08-10/`.
It does not promote positive transfer: fresh learners still win on some
disjoint successors, so the next step is a factual/residual challenger.

That factual challenger is now promoted as a separate external seam:
`ExternalFactoredTransitionModel` keeps the reusable transition base frozen
while `ExternalFactoredTransitionRouter` admits a context-addressed residual
only after held-out one-step, recursive-rollout, and source-retention probes.
The two-seed evidence is in
`session_records/policy_free_factual_residual_growth_promoted_2026-08-10/`.
The residual is one-pass and replay-free in this bounded test; it does not
claim unrestricted computation or general continual learning.

The follow-up factual residual stream admits six regimes plus a reversal into
seven opaque slots with complete-prefix held-out retention, route round-trips,
reversal/missing/corruption controls, exact persistence, and verifier-selected
float16 compression. Both seeds pass while the shared base remains byte-stable;
evidence is in
`session_records/policy_free_factual_residual_stream_promoted_2026-08-10/`.
This is bounded factual-memory scaling, not general continual learning or
arbitrary new computation.

The capacity-scaled follow-up admits nine regimes plus a reversal into ten
opaque residual slots, performs verified `4 -> 8` capacity growth, and uses a
replay-free external reliability gate to allow clean reads while rejecting
corrupted and out-of-distribution evidence. Both seeds preserve the frozen
base and pass held-out prefix retention, persistence, and float16 compression;
evidence is in
`session_records/policy_free_factual_residual_capacity_promoted_2026-08-10/`.
This remains bounded factual-memory scaling, not general continual learning or
arbitrary new computation.

The outcome-only view-routing audit trains `FactorizedOpaqueAddressRouter`
from paired attempted-view outcomes. Across two seeds it reaches `1.000`
held-out route accuracy and `1.000` candidate-permutation accuracy, while the
reward-shuffled control remains at chance and persistent reload preserves the
selected views. This qualifies routing of already-acquired views only; it does
not provide arbitrary new computation or general continual learning.

The four-view scaling audit separates opaque storage identities from controller
queries after operation-derived addresses collide. The joint
`OpaqueAddressRouter` plus paired counterfactual credit reaches `1.000/0.969`
route accuracy across two seeds, with `1.000/0.969` candidate-permutation
accuracy and reward-shuffled routing at `0.215/0.250`. This promotes bounded
four-view routing, not unrestricted memory growth or general continual
learning.

The online-view-growth audit adds a fifth executable view after the four-view
router is frozen. `OpaqueViewRouteExtension` is neutral at creation and is
trained from fresh paired scalar outcomes for the new view. Because a closed
router can be confidently wrong on a novel procedure, the safe selector gives
known routes priority and opens the extension only after an observed failed
opaque old attempt. Two seeds pass five-view behavior, permutation, reload,
corruption, frozen-router, and shuffled-outcome controls with zero replayed
route examples after extension. This is a bounded one-failure cold-start
capability-addition result, not immediate novel-task routing or general
continual learning.

The longer three-step harness composes the same transaction through seven
views and protected float16/int8/int4 replacements. Two independent seeds now
pass the complete boundary with zero replay and frozen controller/earlier
extensions. A historical rejection exposed inconsistent raw-minimum versus
stable-prefix retention accounting; promoted scores and gates now use the
stable-prefix definition consistently. This remains bounded growth evidence,
not unrestricted memory growth or general continual learning.

The multi-step view-growth audit extends that boundary to two sequential
additions. A frozen four-view router first falls back to a new `rotate` view;
after that extension is frozen, a second `complement_rotate` view is opened
only after the old route and first extension both fail. Both new views remain
isolated external artifacts in one physical row. Across two seeds, the full
chain reached `1.000/0.995` routing accuracy with matching candidate
permutation accuracy, `1.000/1.000` first-extension failure discrimination,
zero reward-shuffled new selections, exact reload/corruption protection, and
zero replay after either extension. This is a bounded two-step fallback and
consolidation result; it does not establish unrestricted memory growth,
arbitrary new computation, or general continual learning.

The three-step view-growth audit extends the same boundary to seven opaque
views in one physical row. After the frozen four-view router, `rotate`,
`complement_rotate`, and `adjacent_xor` are acquired sequentially. Each later
extension is opened only after the old route and earlier extensions have been
attempted and failed; earlier extensions and the controller remain frozen.
Across two seeds, the complete chain reached `1.000/0.998` with matching
candidate-permutation accuracy, every prior-extension attempt rate was
`1.000`, shuffled new-view selection was zero, and reload, corruption,
wrong-view causality, frozen-core, frozen-extension, and no-replay controls
passed. This is a bounded three-step growth result, not unrestricted memory
growth, arbitrary new computation, or general continual learning.

`compress_growth_artifact` adds a replaceable caller-owned fixed-capacity
codec. It stores floating growth tensors in float16 and requires an explicit
`allow_dtype_cast=True` at load time; the default loader remains strict. On
the seven-view audit this halves raw tensor payload bytes and preserves every
selected behavior and wrong-view causal gate across two seeds. Storage
compression is behavior-verified outside the controller; it is not learned
new computation or a general continual-learning claim.

`compress_growth_artifact(..., dtype=torch.int8)` provides the stronger
per-tensor symmetric quantization variant with explicit scale entries, and
`decompress_growth_artifact` reconstructs float tensors before execution.
The seven-view audit reduced raw payloads to `0.2506` of float32 size and
serialized artifacts to `0.3278`, while retaining behavior and causal gates
across two seeds. Integer quantization is still a replaceable storage codec,
not learned computation or general continual learning.

`compress_growth_artifact(..., dtype="int4")` adds a packed signed-int4
variant with per-output-row scales and explicit original shapes. Two values
are stored per byte and decompressed before the strict loader. On the same
seven-view audit it reduced raw payloads to `0.1487` of float32 size and
serialized artifacts to `0.2725`, while preserving behavior, causal
wrong-view separation, exact reload, corruption rejection, and frozen-core
gates across seeds `69316` and `69317`. This remains a replaceable storage
codec, not learned compression, arbitrary new computation, or general
continual learning.

The external random-feature transition bank adds a representation-aware
`float16_stats` codec. It keeps the immutable Fourier basis and ill-conditioned
normal matrix lossless, stores the solved predictor in float16, and reconstructs
the target sufficient statistics on restore. It is eligible for promotion only
through the same held-out factual-retention verifier; generic float16/int8
quantization remains available but is not silently substituted for this codec.

`EpisodicContextEncoder` is the next memory-side boundary. It consumes ordered
learned event tensors, opaque actions, scalar outcomes, and presence, then
emits a normalized episode context and per-event credit scores. The
`step()` interface exposes the same recurrent context one event at a time for
online use, without moving that state into the controller. `EpisodicIntentAdapter`
is a zero-initialized external residual that can condition an opaque intention
on that context before a replaceable decoder consumes it. The adapter is
behavior-preserving until trained and carries no task ID or protocol field.
`episodic_context_contrastive_loss` trains context from paired augmented
episodes without task labels; `paired_event_credit_loss` trains event credit
from common-random scalar intervention outcomes. In the promoted bounded
audit, this context beat pooled-event routing `0.9688/1.000` versus
`0.500/0.500`, enabled a fresh no-replay route append, and passed ablation,
permutation, retention, shuffled-outcome, and credit gates across two seeds.
This remains replaceable external state, not general continual learning.

`ExternalCapabilityProgram` packages that boundary into one replaceable
memory-side program: its recurrent context state and intention adapter are
owned outside the shared controller, while a caller attaches any compatible
decoder on the output bus. In the promoted two-seed artifact-bank audit, three
independent programs (`reverse4`, `forward4`, and `complement4`) were acquired
from fresh rendered episodes, selected with opaque learned route evidence,
persisted, reloaded, and causally separated from the wrong program.
Stable-prefix bits, parent retention, shuffled-route, permutation, corruption,
and frozen-core gates all passed. This is a bounded controller-as-CPU /
memory-as-files capability result. The follow-on append audit fills a
capacity-three bank with protected rows, rejects a fourth write rather than
evicting a mastered program, grows the bank transactionally, and acquires a
`rotate4` artifact through a fresh route extension while the parent and old
router remain frozen. Both seeds pass four-program retention, routing,
permutation, reload, corruption, causal wrong-artifact, and zero-replay gates.
This is still one protected append, not repeated open-ended growth, arbitrary
program induction, or general continual learning.

`ExternalCapabilityPipeline` is the canonical variable-length composition
boundary for those programs. It keeps one recurrent state per memory-side
program and serially passes the opaque intention from one adapter to the next;
an empty pipeline is an exact identity. Programs must share only the versioned
event/action/intention dimensions, so stacking or replacing them does not
resize the controller or add a task-specific reasoning branch. This is the
execution foundation for testing reusable program composition; the pipeline
itself does not claim that independently learned programs will solve an
arbitrary novel composition. Its optional `head_only` event-visibility mode
gives the first program the learned event while later programs receive only
the prior opaque intention, opaque feedback, and scalar outcome. That mode is
a diagnostic and execution contract for detecting raw-event shortcutting; it
does not assign semantic meaning to the intermediate intention.

`EpisodicCreditHead` isolates event-credit state for one external capability.
The two-step follow-up trains one fresh head per appended procedure while the
shared context encoder and earlier credit heads remain frozen. Across two
seeds, both sequential routes recovered at `1.000/1.000`, old-route retention,
prior-extension attempts, new-route ablations, shuffled-outcome controls, and
per-extension credit localization all passed with zero replay. This is a
bounded growth result; learned eviction, nonstationary discovery, and general
continual learning remain open.

`CapabilityRetentionLedger` is the memory-side protection layer for the next
continual-learning step. It consumes only an opaque learned address and a
scalar verifier outcome, tracks stable mastery across every measured prefix,
and masks protected rows from both `ContentAddressedMemory` and
`ExecutableArtifactMemory` eviction. A single noisy failure does not release a
mastered capability; sustained low outcomes trigger an explicit reversal and
start a fresh mastery era. If every occupied row is protected, writes raise a
growth/consolidation signal instead of silently forgetting a capability. The
ledger persists beside disk-backed memory and is copied through artifact
compaction, and canonical runtime checkpoints.

`evaluate_retention_gate` supplies the corresponding transactional promotion
check: a new capability must pass its stable prefix threshold while every
already-retained verifier score stays above the declared floor. This is the
first implementation of retention-safe memory growth in the canonical runtime,
but it is not yet a Brain Workshop mastery result or a claim of general
continual learning; the next audit must connect the ledger to a replay-free
multi-rung Brain Workshop extension and measure reversal behavior.

The public boundary is exposed from `neural_computer.__init__`. Component
checkpoints are loaded into caller-constructed encoders, controller, memory,
and decoders through `load_runtime_components`; checkpoint metadata never
constructs an implicit modality branch.

For a keyboard-backed agent, the normal execution loop is therefore:

```python
feedback_action = keypress_encoder(previous_key_index)
feedback = ControllerFeedback(action=feedback_action, ...)
runtime_output, state = runtime.step_events(events, state, feedback)
keypress = keypress_decoder.decide(runtime_output.intention)
```

The decoder owns sampling and propensity accounting. The controller never
receives raw key codes and never emits device-specific key codes.

The controller also emits an execution-plane policy with three operational
states: `WAIT` keeps the intention tentative while transport may provide more
events, `THINK` spends a bounded quiet recurrent tick, and `COMMIT` releases
the current opaque intention to the output bus. A learned, age-gated timeout
residual can make a second decision after `WAIT` when evidence remains absent.
The controller also includes a bounded zero-initialized pairwise event-attention
residual for learned cross-token binding, plus versioned feedback/source-key
interactions for outcome-conditioned evidence binding. Runtime v28 carries the
generic learned source-credit policy: prior event tokens, opaque source keys,
and feedback produce a trust-space credit vector, gated by normalized source
attribution and averaged over present tokens before updating persistent source
trust. Its output bias is neutral at initialization, avoiding an unconditional
source preference and making the update scale independent of encoder count.
This is still one controller; the runtime only enforces the deliberation bound
and does not add a reasoning module.

Runtime v28 exposes a payload-only latest-event memory address with a residual
learned-event identity path that remains stable across recall age and
irrelevant prior events. The write policy receives retained latest-token pair
context plus average and strongest current-to-prior matches, so utility
decisions can condition on bounded event interference without a
modality-specific branch. During outcome-only training, v28 can optionally
sample a Bernoulli write decision
with a straight-through differentiable transaction and expose its opaque
log-probability for policy-gradient credit. This is training infrastructure,
not a deployed protocol. The pre-v22
cue-guided retention diagnostic remains rejected; the corrected query-cue and
latest-address qualification is recorded separately. The v73 outcome-only
retention/transfer rung is promoted for the narrow verifier after three-seed,
four-pair unseen-token, causal, persistence, and positive fresh-transfer
controls. The v76 three-seed v27 outcome-only three-slot/two-row retention
rung also passes balanced-position, unseen-token, causal, persistence, and
checksum controls. These results do not establish general episodic memory or
natural-modality capability.

Memory addresses use one shared learned projection plus a residual learned
event-identity path over the latest learned event payload for both writes and
reads. Transport metadata such as event age,
duration, timestamp presence, and confidence remains available to reasoning
and write utility, but cannot make the same event address differently at
recall time. v23 checkpoints migrate with their transport-augmented address
behavior; v24 checkpoints migrate with the feedback residual disabled, and v25
and v26 checkpoints migrate with their prior address behavior. New checkpoints
are v28. The controller also exposes an optional generic growth-register
chain: slot weights are independently loadable artifacts, slot state lives in
controller state only while executing, prior-only slots receive only the
preceding learned register, and recurrent consumer registers remain outside
the frozen core. This is the canonical CPU-like execution boundary for
externally stored learned factors; it does not assign semantic names to
registers or claim arbitrary program synthesis. An optional `growth_gated`
variant adds a zero-initialized learned contribution gate over each opaque
register, while `growth_recurrent_from=0` gives the first external slot its
own temporal state. These extensions are behavior-preserving until trained;
`growth_from_intention=True` exposes the frozen processor's learned intention
to the external slot, and `growth_gate_from_context=True` lets the gate use
the current opaque context. The matched transfer audit promotes this
parent-conditioned recurrent boundary across two seeds (`1.333x--1.667x`
fresh-over-inherited stable-bit transfer) while retaining the parent. This is
still a narrow transfer result, not general continual learning.

The package also exposes `ExternalCapabilityRegisterMachine`. It is a
shared learned interpreter over an external working register: capabilities
are opaque instruction vectors in a variable-length serial chain, not new
controller branches. A recurrent external context reads each standardized
learned event, opaque feedback record, and controller intention; downstream
instructions receive only the preceding register. The rendered reverse→
complement audit promotes a narrow `2.0x` fresh-over-inherited stable-bit
composition transfer across two seeds with retention, shuffled, missing,
reload, corruption, frozen-core, and zero-replay controls. It remains a
bounded result, not arbitrary program induction or general continual learning.

Memory is a replaceable `MemoryBackend` v1 contract. The default
`ContentAddressedMemory` keeps a bounded content-addressed index in the
runtime, while `PersistentContentAddressedMemory` atomically snapshots and
checksums the same learned keys, values, strengths, timestamps, and version to
disk. Query alignment remains differentiable through read scores and weights;
inside an explicit training transaction, pending values also expose a
differentiable write-strength gate while persisted rows stay detached state.
Query, read, and write-receipt records
are schema-validated, matching keys are upserted instead of duplicated, failed
durable writes restore the prior in-memory state, and runtime checkpoint loads
validate and roll back memory components as well. A narrow scalar outcome-recall
rung now passes clear-memory, corruption, persistent-replacement, four-pair
unseen-token, and matched fresh-transfer controls for the narrow verifier. The
corrected v76 three-slot/two-row rung also passes its balanced-position,
unseen-token, causal, and persistent-memory gate; this remains a narrow
outcome-only claim rather than general episodic memory.
For batched independent trajectories, an optional opaque `memory_scope` selects
one of fixed-capacity isolated banks without entering the learned key/value
content; the legacy single-scope layout and checkpoints remain compatible.

`MemoryCandidates` exposes detached opaque candidate rows for a replaceable
memory-side policy, and `target_index` lets that policy request generic row
replacement without adding physical-slot semantics to the controller. The
`ExternalMemoryEvictionPolicy` qualification uses this interface with paired
scalar counterfactual outcomes and a frozen controller. It promotes only a
narrow learned-utility eviction mechanism for the audited synthetic bank; the
memory backend still does not claim general episodic utility or unrestricted
learned memory growth.

`AppendOnlyContentAddressedMemory` and its persistent replacement provide the
next growth boundary. They append unmatched learned keys without changing
controller shapes, upsert matching keys, and serialize variable-length state
with checksums. The frozen-controller growth audit passes 64, 256, and 1,024
opaque records across two seeds with permuted exact recall, zero clear-memory
hits, and persistent recovery. This qualifies logical storage growth, not
learned compression or general continual learning.

For a bounded external policy, `MemoryCandidates.pad_to_capacity(n)` exposes
the current append-only rows in a fixed-width candidate view. Added rows are
zero-filled and explicitly unoccupied; the view does not resize the backend or
the controller, and any rewrite still requires the backend's verifier-gated
transaction boundary.

The original v74 three-slot/two-row rung is retained as a superseded harness
record because its duplicated counterfactual arms did not preserve balanced
target positions. The corrected v76 rung now qualifies bounded learned
multi-row retention. The v75 synthetic cross-adapter rung qualified the
two-row neural-IR case. The v77 three-seed rung now also qualifies three-row
outcome-only retrieval with an opaque target cue, cued-row-last presentation,
and persistent reload/recovery; all fresh-token and swapped-slot controls
pass. The three-slot/two-row bounded-interference variant also passes after
separate strict write and learned-IR read-match thresholds were introduced;
natural-modality alignment and broader episodic utility remain open. Evidence is in
`session_records/cross_adapter_memory_amodal_v77_2026-08-04/`.

The v78 cross-adapter follow-up also randomizes the target position within a
three-row sequence while retaining only two memory rows. The generic
counterfactual write-utility trainer stabilizes fresh-reader minima at
`0.991/0.988/0.998` across seeds, with persistent and memory-corruption
controls passing. This is still a bounded synthetic outcome-only result;
cue-conditioned utility, capacity-one compression, natural modalities, and
general episodic memory remain unqualified. Evidence is in
`session_records/cross_adapter_memory_amodal_v78_2026-08-04/`.

## Adaptive reader capacity growth

`AdaptiveOnlineEpisodicRelationReader.expand_capacity()` is the memory-side
growth boundary. It creates a larger replacement reader without changing the
controller or previously allocated slots. Learned weights may be copied when
the candidate is already useful; a fresh larger reader is allowed only for an
unmastered candidate whose new opaque outcome evidence justifies replacement.
The canonical audit promotes this failure-triggered reset-and-grow transaction
from capacity five to six across three seeds, with old-capability retention,
route reload, causal controls, frozen-core, and zero-replay gates. This is a
bounded growth contract, not unrestricted memory growth or arbitrary new
computation.

The growth transaction resets a complete unmastered external slot when fresh
failure justifies replacement. Resetting only the adaptive reader is unsafe:
the intention adapter and keypress decoder can already encode a failed route.
`CanonicalBrainWorkshopAgent.expand_adaptive_relation_capability(...,
reset_failed_reader=True)` therefore replaces the reader, adapter, route
scorer, opaque key, and decoder together, while mastered slots and the
controller remain untouched. The recursive Brain Workshop audit promotes two
such capacity transactions, `5 -> 6 -> 7`, across three seeds with zero
replay. This remains a bounded growth contract, not unbounded memory or
arbitrary new computation.

## Retention-safe capability eviction

`CanonicalBrainWorkshopAgent.replace_unprotected_adaptive_relation_capability()`
is the bounded-bank lifecycle boundary. It refuses to evict a protected
capability, clears stale global and context-conditioned route evidence, and
replaces the complete unprotected external slot. The three-seed canonical
audit promotes replacement of an unmastered slot while preserving two
mastered capabilities, restoring route discovery, surviving reload, and
keeping the controller frozen with zero replay. This is retention-safe slot
reuse, not yet learned general utility, consolidation, unbounded memory
growth, or general continual learning.

`ExternalCapabilityEvictionPolicy` is the memory-side learned utility boundary.
It consumes an incoming learned event tensor plus detached opaque capability
addresses and scores which unprotected candidate is most disposable. Fresh
scalar verifier outcomes train the policy; outcome summaries, task IDs, raw
modalities, and physical protocol formats do not enter the policy input. The
retention ledger remains an independent safety mask, so learned utility cannot
evict mastered state. The canonical three-seed audit promotes this narrow
context-conditioned utility policy with controller freezing, chance-level
reward-shuffle and feature-corruption controls, stable replacement, stale-route
reset, and zero replay. It does not claim general episodic utility,
consolidation, unbounded memory, or general continual learning.

The episodic context-credit harness supports a contiguous sequence of external
additions through `--new-families`. The four-step audit freezes the shared
context and old router, gives each new capability an isolated
`EpisodicCreditHead` and route extension, and requires later capabilities to
attempt every earlier extension before activation. It passes route,
permutation, retention, causal-ablation, isolated-credit, and zero-replay
gates across two seeds. This is bounded replay-free external growth, not yet
unbounded memory, learned consolidation, arbitrary program induction, or
general continual learning.

The same harness supports a longer five-token pattern bank and eight
sequential additions. The promoted eight-step audit preserves the frozen old
route and isolated credit state across all additions with causal extension
ablations, candidate permutation, reward-shuffle rejection, and zero replay.
The short-budget control fails old-route retention, so context and route
training budgets are scaled with temporal length. This remains a bounded
continual-memory result, not general continual learning.

The pattern bank can also be generated from an explicit episode length. The
promoted length-six audit provides 20 same-statistics procedures and eight
sequential additions with isolated route and credit state. Longer histories
require a larger frozen-context/router acquisition budget; the seed-unstable
under-budget control remains rejected evidence.

The generated length-six audit is also composed with the opaque retention
ledger: all ten capabilities initially protect, a fully protected bank refuses
eviction, only the newest capability reverses after sustained failures, and it
re-protects after fresh successful outcomes. Both canonical seeds pass the
retention, reversal, recovery, causal, and zero-replay gates. This is a
bounded retention-safe growth contract, not learned consolidation, unrestricted
memory growth, or general continual learning.

The artifact store now enforces the same boundary during compaction. Generic
compaction refuses to drop protected rows; consolidating protected rows requires
fresh candidate outcomes; and an accepted replacement persists those outcomes
as new retention evidence. The two-seed artifact audit preserves behavior,
opaque aliases, reload integrity, and the frozen core with zero consolidation
updates and zero replay. This is retention-aware logical compaction, not learned
byte compression or general continual learning.

The canonical package now also exposes `OpaqueConsolidationPolicy`. It scores
unordered pairs using only controller-native keys, values, strength, and
relative age; `verify_consolidation_proposal` applies an immutable snapshot
rewrite and requires an independent behavior verifier plus optional retention
gate before adoption. The two-seed audit learned duplicate-pair selection,
preserved candidate permutation, and composed four accepted rewrites from
eight rows to four with zero replay. This is a promoted memory-side mechanism,
not yet executable-artifact behavioral consolidation or general continual
learning.

`ExecutableArtifactMemory.consolidate_verified()` also accepts a
`candidate_outcome_probe`. It builds the candidate in a separate directory,
lets the caller measure fresh opaque verifier outcomes, applies the retention
gate, and only then invokes the adoption verifier. This is the required
transaction path when source rows are already protected; it preserves source
immutability without substituting synthetic retention outcomes.

The promoted executable route rung composes this with the generic
`OpaqueAddressRouter`: a learned query selects an opaque alias candidate,
`ExecutableArtifactMemory.promote()` resolves it, and the caller loads the
verified view through the frozen controller. The router is permutation
equivariant and outcome-trained; it does not receive operation names or
protocol fields. This qualifies bounded learned address acquisition, not
general skill induction or unrestricted continual learning.

The online view-growth harness now composes that route with the retention
ledger. Four old executable views are protected from fresh outcomes; a fifth
view is constructed and probed in a disposable transaction; and adoption
requires both a new-candidate retention floor and independent old/new behavior
verification. The promoted two-seed audit retains the frozen controller and
old router, learns the new route without replay, and preserves five opaque
views in one physical row. This is a bounded retention-safe online addition,
not yet unrestricted growth or general continual learning.

The same transaction composes across two sequential additions in the
multi-step harness. The intermediate replacement is required to become
protected before the next candidate is built; the second transaction must
preserve all earlier behavior and the final row must retain six opaque views.
Two seeds pass this bounded two-step audit with zero replay after either
addition. This establishes composition of the safety boundary, not open-ended
growth or general continual learning.

`ExecutableArtifactMemory.grow()` provides the corresponding finite-capacity
escape hatch: protected eviction fails explicitly, then a new verified bank
can copy live artifacts, aliases, strengths, and retention records without
mutating the source. The new bank can admit a capability after growth. This
is a storage safety primitive, not learned capacity planning or general
continual learning.

`OpaqueCapacityPlanner` is the next replaceable memory-side boundary. It
learns to choose generic admission, safe eviction, verified consolidation, or
capacity growth from an incoming learned key/value and permutation-equivariant
opaque bank summaries. Retention masks and behavior verification remain
outside the learned selector. The two-seed capacity-planning audit promotes
the bounded action mechanism and protected-bank growth path with zero replay;
it does not establish unrestricted memory growth, arbitrary computation, or
general continual learning. Evidence is in
`session_records/sequence_working_memory_2026-08-02/artifact_capacity_planning_v1_2026-08-06/`.

The canonical working-memory growth harness now uses the same lifecycle for
producer→consumer composition. Two independently acquired growth artifacts
are transactionally replaced by one namespaced row; a fresh runtime reloads
the row and must pass the held-out behavior verifier before adoption. Two
seeds pass the one-row, exact-reload, frozen-core, producer-ablation,
prior-read-ablation, missing-sequence, and reward-shuffled gates with zero
replay. This establishes a narrow protocol-agnostic external composition
boundary, not arbitrary program synthesis or general continual learning. The
evidence is in
`session_records/sequence_working_memory_2026-08-02/canonical_growth_pressure_lifecycle_composition_v1_2026-08-06/`.

The external route/credit boundary has also filled the closed generated
length-six pattern bank: two old capabilities plus 18 sequential additions.
Across two seeds, all 20 opaque routes retain permutation, causal, reversal,
recovery, full-bank protection, and zero-replay gates. This is a bounded
retention result; the remaining challenge is a changing distribution and
dynamic expansion beyond a fixed family bank.

The first changing-distribution audit freezes length-six capabilities and
appends length-seven capabilities without replay. Old routes, shifted credit,
causal additions, reversal/recovery, and retention remain intact, but a
cross-seed reward-shuffled false positive rejects the rung. The memory boundary
is therefore ahead of shifted credit calibration; the rejected evidence is in
`session_records/sequence_working_memory_2026-08-02/nonstationary_growth_episode6_to7_rejected_v1_2026-08-06/`.

The corrected shift harness uses an antithetic trainer-only null for shuffled
outcomes. It now passes the length-six→length-seven distribution shift across
two seeds with old-route retention, new-route causality, shifted credit,
reversal/recovery, and zero replay. This qualifies one controlled
nonstationary boundary while preserving the rejection record for the weaker
random-shuffle control.

The full-bank follow-up passes as well: 18 fresh length-seven capabilities
fill the 20-family bank after a frozen length-six base across two seeds. This
is the current bounded nonstationary-growth ceiling; repeated distribution
shifts and dynamically expanded banks remain open.

The length-six→length-eight full-bank audit also passes across two seeds,
showing the boundary tolerates a larger temporal shift. The weakest route is
`82.81%`, so repeated shifts and dynamic expansion—not another adjacent-shift
smoke—are now the meaningful next tests.

The length-six→length-ten full-bank shift is the current rejected frontier:
one seed leaves a shifted capability below mastery before reversal. The
correct response is confidence-aware acquisition/retention, not lowering the
protection threshold.

The repaired 6→10 rung passes both seeds after doubling shifted-extension
acquisition and stabilizing the antithetic null. This keeps the protection
threshold hard while making acquisition depth explicit in the evidence; the
128-update failure remains a regression control.

The repeated-shift boundary now passes length six→eight→ten in one frozen
run, preserving earlier routes and credit while adding 18 capabilities with
zero replay. This is the strongest bounded continual-growth evidence; the
next architectural pressure is a dynamically expandable bank and repeated
shifts beyond the closed 20-family generator.

The next three-shift audit crosses that ceiling: length six→eight→ten→twelve
adds 8, 10, and 12 capabilities in sequence, for a 32-capability bank. Across
two seeds, phase route floors are `89.06%/92.19%`, `89.06%/90.63%`, and
`85.94%/92.19%`; old-route/permutation, causal credit, full-bank protection,
isolated reversal/recovery, antithetic null, and zero replay all pass. This
qualifies dynamic external-bank growth beyond 20 capabilities. It remains a
finite generated-family result with externally trained route/credit acquisition,
not unbounded memory growth or general continual learning. Evidence is in
`session_records/sequence_working_memory_2026-08-02/repeated_shift_growth_6to8to10to12_v1_2026-08-06/`.

The same 32-capability schedule now supports a copy-on-write average prior for
new route adapters. Both seeds pass all hard gates at 256 updates per family,
but the effect is not uniformly positive versus fresh initialization: one
seed improves its final-shift floor while the other loses a small amount on
the final shift. The 128-update control fails both seeds at the final shift,
so prior reuse is promoted as safe state reuse, not as a reliable sample-
efficiency gain. Evidence is in
`session_records/sequence_working_memory_2026-08-02/repeated_shift_growth_6to8to10to12_growth_prior_v1_2026-08-06/`.

`OpaqueCandidateGrowthRouter` is the next reusable memory-side boundary. One
candidate-conditioned, permutation-equivariant router can score a variable
opaque candidate bank, avoiding a new learned extension module for every
capability. The promoted 6→8→10→12 audit uses one router per shift and a
learned trajectory-statistics query. Across two seeds it grows to 32
capabilities with phase floors `0.9844/0.9375`, `0.9844/0.9688`, and
`0.9531/0.9375`; causal, retention, reversal, null, and zero-replay gates
pass. The corrected strict sequential operational route-permutation audit is
`0.9906/0.9911`; the earlier `0.4932/0.4943` reading was a harness false
negative caused by comparing a remapped physical row to its unpermuted family
index. This is a bounded reusable-growth result, not general continual
learning, and acquisition currently requires 16,384 route updates per
expansion. See
`session_records/sequence_working_memory_2026-08-02/shared_growth_router_6to8to10to12_trajectory_stats_v1_2026-08-06/`.

The shared-router acquisition budget has since been cut to 8,192 updates per
expansion. Both seeds still pass the full 6→8→10→12, 32-capability audit with
phase floors `0.9844/0.9844`, `0.9688/0.9063`, and `0.9219/0.9063`, operational
permutation `0.9875/0.9802`, causal credit, retention, reversal, null, and
zero-replay gates. Total optimizer updates fall 46.9% versus the previous
rung. This promotes acquisition efficiency for the existing bounded policy;
the fixed trajectory query and random opaque-key association remain open.
See
`session_records/sequence_working_memory_2026-08-02/shared_growth_router_6to8to10to12_trajectory_stats_8192_v1_2026-08-06/`.

## External compute-candidate screening

`ExternalComputeCandidateScreen` packages the persistent opaque route ledger
as a compute-library admission aid. It indexes learned event/context queries,
records only attempted opaque candidate indices and deterministic scalar
outcomes, and returns a learned-first trial order. The screen is deliberately
not an admission policy: a fresh verifier floor must still be passed by the
candidate that is actually tried. This lets a later experiment measure trial
and latency savings without allowing a stale screen to create a capability
claim. Its state is versioned and reloadable through
`neural-computer.external-capability-compute-screen.v1`.

`LearnedComputeCandidateScreen` adds a factorized query/key scorer for novel
contexts. It is explicitly disabled until external fresh evidence enables it,
then learns pairwise compatibility from attempted scalar outcomes. A paired
six-candidate audit routes novel contexts over known candidates at `1.0000`
across two seeds versus `0.2500` cold-start, with exact candidate permutation,
reload, frozen-core, and reward-shuffled null controls. Outcome-unseen
candidates remain a deliberate cold-start failure. Evidence is in
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_screen_opaque_novel_context_replicated_promoted_v1_2026-08-07/`.

A direct fresh-outcome calibration of two unseen candidates acquires them at
`1.0000` but collapses known-candidate routing to `0.2083/0.2500`; this is a
rejected shared-screen mutation. The next design must append isolated screen
state and preserve the frozen prior screen.

The append-only replacement now passes the same two-seed pressure test:
known routing remains `1.0000`, unseen routing is `0.0000` before the scalar
failure gate and `1.0000` after 64 fresh updates, with permutation, reload,
frozen-core, reward-shuffled, and zero-replay controls passing. This promotes
safe external screen growth only; it is not a claim of unrestricted memory
growth or general continual learning. Evidence is in
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_screen_append_only_promoted_v1_2026-08-07/`.

The append boundary also passes a two-stage audit: four unseen candidates are
split across two isolated extensions, later activation requires cumulative
failure of the base and earlier stage, and both seeds reach `1.0000` unseen
routing from `0.0000` pre-activation while known routing remains `1.0000`.
Stage-local permutation, reload, frozen-core, reward-shuffled, and zero-replay
controls pass. This remains bounded external growth, not general continual
learning. Evidence is in
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_screen_append_only_two_stage_promoted_v1_2026-08-07/`.

The boundary is now cardinality-independent: multi-candidate stages retain
pairwise ranking, while singleton stages use attempted-outcome calibration
from fresh positive and negative verifier attempts. A mixed `[1, 2]` two-stage
audit reaches `1.0000` unseen routing from `0.0000` pre-activation across both
seeds with retention, permutation, reload, null, and zero-replay controls
passing. Evidence is in
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_screen_cardinality_independent_mixed_promoted_v1_2026-08-07/`.

The same mixed audit passes at 128 fresh calibration updates per stage,
halving append calibration cost. Blindly inheriting the base screen weights
is retained only as a rejected control because it harms new-stage acquisition;
fresh extension initialization remains canonical. Evidence is in
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_screen_cardinality_independent_mixed_128_promoted_v1_2026-08-07/`.

Selective query-side transfer improves the same mixed audit again: copying
only query projections and router query encoders passes both seeds at 64
updates per stage, while fresh initialization fails one matched seed at that
budget. Candidate-key and matching state remain fresh. Evidence is in
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_screen_selective_query_prior_promoted_v1_2026-08-07/`.

Three sequential singleton append stages now pass both seeds at 256
calibration updates per stage with the selective prior, while 64 fails both
seeds and 128 fails one. A matched fresh control also passes at 256, so this
is replicated bounded three-stage growth rather than a new three-stage prior
efficiency claim. The immediate bottleneck is stage-wise calibration cost as
depth grows. Evidence, accounting, and lower-budget controls are in
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_screen_append_only_three_stage_boundary_v1_2026-08-07/`.

Append-only prior initialization now accepts a validated strength in `[0, 1]`.
Half-strength query transfer repairs the three-stage 128-update failure but
is not a universal default: it loses one seed on the earlier two-stage
64-update boundary. This keeps prior selection outside the frozen controller
and makes negative transfer measurable rather than silently baked in. See
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_screen_prior_strength_boundary_v1_2026-08-07/`.

The append screen now also passes a fourteen-candidate, three-stage bank at
32 updates per stage, with six new candidates split two per isolated stage.
Both fresh and query-prior controls pass, so this is evidence of bounded bank
scaling rather than a transfer advantage. The next pressure test is deeper
sequential growth, where calibration cost and retention must compound without
replay. See
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_screen_append_only_bank14_three_stage_32_promoted_v1_2026-08-07/`.

The same two-candidate stage contract now reaches five isolated stages in a
twenty-candidate bank at 32 updates per stage. Fresh and query-prior controls
both pass, so the result is repeated bounded growth, not prior efficiency.
The next architectural boundary is safe compaction of these external screen
extensions. See
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_screen_append_only_bank20_five_stage_32_promoted_v1_2026-08-07/`.

`AppendOnlyLearnedComputeCandidateScreen.consolidate_verified` now provides
that compaction transaction: it accepts a caller-trained replacement for
consecutive stages, preserves logical candidate count, never mutates the
source, and adopts the smaller physical extension set only when a fresh
behavior verifier passes. It is a boundary contract, not evidence that
learned consolidation already works.

The first fresh-outcome plus source-distillation compaction audit is
intentionally rejected: strict per-candidate retention failed on both seeds,
and a naïve copied-stage replacement was rejected. The next task is repairing
source mastery and replacement training before any compression claim.

The pairwise router control is likewise rejected: it underperforms the
factorized router at the five-stage bank-20 rung and is not part of the
canonical API. The active bottleneck is upstream candidate-signature
separation and per-candidate mastery.

The candidate-screen harness now records per-target top-1 mastery and
candidate-key separation diagnostics. Aggregate routing no longer promotes an
append boundary when one target is hidden by the mean: the strict gate requires
every audited target to clear the mastery floor. The bank-20 audit shows why:
one seed has aggregate unseen accuracy `0.8958` but a target at `0.0`, while
the twenty-key representation has nearest-neighbor cosine `0.9982` and
effective rank `3.59`. The next implementation target is therefore a richer,
behaviorally grounded candidate signature, not another router variant.

A spatial-binding frontend control shows that separation alone is not enough:
it raises key effective rank from `3.59` to `6.16` on the weaker seed, but
unseen routing falls from `0.8958` to `0.8021`. The next experiment must train
the event/key alignment or its behavioral trace jointly; simply preserving
more frontend geometry is not a sufficient fix.

Full append-prior transfer is also insufficient as a general remedy: it brings
the unseen extension to `1.0000/1.0000`, but the weaker seed retains source
known-target holes at `0.7` and `0.0`. The next high-ROI intervention is source
screen mastery and query/key alignment, with append initialization treated as
solved at this bounded rung.

The source-mastery intervention now passes the strict twenty-candidate,
five-stage boundary across both seeds at 1024 source updates with full append
priors. A matched fresh-extension control fails the hard seed on one unseen
target, so the prior is retained as a robustness mechanism. This is a bounded
growth result; the hard seed's key effective rank remains `3.59`, leaving
representation/alignment as the next bottleneck.

Doubling local append calibration does not change the spatial control's hard
seed: it remains `0.8125` with two per-target holes. More extension updates
are therefore not the next move; the alignment must adapt the candidate key to
the observed behavioral evidence or use a better learned event signature.

The frozen signature-normalizer contract was tested and rejected as a global
replacement: it repairs source mastery but harms unseen acquisition, and a
dual raw-plus-normalized view is worse. Future representation diversity must
be page-local and verifier-selected, not concatenated into one undifferentiated
address space.

The page-local ABI now passes the matched bank-26/six-stage pressure test with
the source page on a frozen affine-normalized view and append pages on raw
identity views. A local rank-margin gate keeps page score scales independent;
both seeds reach strict `1.0000/1.0000` known and unseen per-candidate mastery,
with permutation, reload, null, frozen-base, frozen-core, and zero-replay gates
passing. The promoted budget is 1,024 normalized source updates, 512 raw-prior
updates, and 32 fresh updates per append stage. A 512/512 source split fails
hard-seed known retention at `0.8854`, so this is a structural bounded-growth
promotion with a measurable source-budget cost, not yet learned representation
selection or general continual learning. See
`experiments/compute_candidate_screen_amodal/train_page_local.py` and the
promoted/rejected session records.

At 46 candidates, the same page-local boundary retains all 26 unseen rows but
fails strict source retention at `0.9271/0.4000` on both seeds. Normalized key
rank is materially higher (`11.90/12.87`), so the next bottleneck is source
screen capacity/interference rather than another global signature transform.
Increasing scorer latent width from 32 to 64 is also rejected: known floors
are `0.0000/0.4000` while unseen remains perfect. The next capacity test is
router hidden width or source competition isolation. Hidden width 128 is also
rejected (`0.8542/0.8021` known, zero per-target floor), so source competition
isolation is now the preferred intervention.

The source-sharding intervention passes the 46-candidate rung: two normalized
source pages of ten candidates each plus raw unseen append pages reach strict
`1.0000/1.0000` known and unseen mastery on both seeds. It reduces the audited
cost to 2,496 optimizer updates and 477,696 verifier bits, versus 3,008 and
1,423,872 for the unsharded page-local control. This is a promoted bounded
source-competition and sample-efficiency result; page retrieval is still
physical-order based and general continual learning remains unqualified.

The same source sharding scales to 64 candidates with three normalized source
pages of ten and 17 raw unseen pages: both seeds pass strict `1.0000/1.0000`
known and unseen mastery plus permutation, null, reload, page-immutability,
and zero-replay controls. This is a larger bounded isolation result, not yet
learned page retrieval or general continual learning.

The learned page-router audit closes the next sub-boundary. A frozen external
router receives a learned query and opaque page summaries, then learns from
scalar verifier outcomes produced by attempting each local page. At 64
candidates, both seeds retrieve all three source pages and all 30 source
candidates at `1.0000`, survive page-order permutation and exact reload, fail
the reward-shuffled null, and leave the controller byte-identical with zero
replay. This promotes bounded learned page addressing, not arbitrary memory
growth or general continual learning. See
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_page_router_64_promoted_v1_2026-08-07/`.

The append-only control rejects simply freezing that router and concatenating
new page keys: 34 new candidates in 17 pages fall to `0.3958` and `0.1250`
candidate accuracy across the two seeds. This is the current memory-growth
blocker. The next mechanism is a separately trainable append-router overlay,
with verifier-gated fallback from the frozen source router; see the rejected
control at
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_page_router_append_frozen_rejected_v1_2026-08-07/`.

The token-preserving append-router overlay now passes the growth rung. The
frozen source router handles the original pages; scalar verifier failure gates
an append router that keeps every normalized opaque candidate token rather than
using a lossy page mean. At 64 candidates, both seeds reach strict `1.0000`
candidate/page and per-target/per-page mastery with permutation, shuffled-null,
frozen-source, unchanged-controller, and zero-replay controls. The cost is
10,816 optimizer updates and 1,887,744 verifier bits per seed. This is bounded
no-replay external page addressing, not general continual learning. Evidence:
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_page_router_append_token_overlay_64_promoted_v1_2026-08-07/`.

The cost boundary has since dropped to 3,072 updates per append router. Both
seeds retain strict `1.0000` candidate/page and per-target/per-page mastery with
all permutation, shuffled-null, frozen-source, unchanged-controller,
verifier-fallback, and zero-replay controls. This costs 8,768 optimizer updates
and 1,469,952 verifier bits per seed. See the superseding record:
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_page_router_append_token_overlay_64_promoted_v2_2026-08-07/`.

The overlay also survives two append generations: 30 source candidates plus
two independent 18-candidate generations reach strict `1.0000` across 66
candidates and 21 pages on both seeds. Each generation uses its own scalar-
outcome-trained token router; verifier failure cascades from source to the
later generation. The audit passes permutation, generation-local shuffled
nulls, frozen source/controller, no unresolved rows, and zero replay. See
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_page_router_two_generations_66_promoted_v1_2026-08-07/`.

The external boundary now includes an appendable, independently freezeable
`LearnedOpaqueCandidateKeyMemory`. Its first outcome-trained update policies
are rejected: joint base/key updates cause interference, while extension-only
updates preserve behavior but do not beat static keys. The contract is useful
for future behavioral memory, but no address-update policy is promoted yet.

At the next capacity rung, 26 candidates across six append stages still acquire
all unseen rows, but the source screen loses strict mastery (`0.9271/0.7500`
known routing). This is the current capacity boundary: source interference
must be solved before claiming scalable continual growth.

Doubling the bank-26 source budget again to 2048 does not repair replication:
one seed still has known-target holes despite perfect unseen acquisition. More
updates are therefore not the current lever; representation and source-router
interference are.

The next 46-capability fourth-shift pressure test is intentionally rejected:
late length-14 rows fail stable protection at hidden 256/8,192 updates,
hidden 512, and an adaptive late-budget control. This is evidence for
confidence-aware targeted acquisition and capacity planning, not a reason to
weaken the retention gate. See
`session_records/sequence_working_memory_2026-08-02/shared_growth_router_6to8to10to12to14_46caps_rejected_v1_2026-08-06/`.

The matched modular route/credit boundary now passes the same fourth shift
with 46 total capabilities when each new route and credit head is isolated
from earlier learned state. Across seeds `69316` and `69317`, final-shift
route floors are `0.8594` and `0.8750`; old-route retention, permutation,
causal extension, null, full-bank protection, reversal/recovery, and
zero-replay gates all pass. This localizes the shared-router 46-capability
failure to interference in one mutable candidate scorer, not to external
memory capacity. The modular result remains a bounded externally trained
route/credit mechanism, not general continual learning. See
`session_records/sequence_working_memory_2026-08-02/repeated_shift_growth_6to8to10to12to14_isolated_v1_2026-08-06/`.

The 46-capability modular boundary also persists its external route and
credit state independently through `PersistentOpaqueStateStore`. Across both
seeds, 89 state files reload with exact route/credit behavior and deliberate
checksum corruption is rejected; persistence adds no verifier bits,
optimizer updates, or replay. This closes the durable-state boundary for the
bounded result while leaving unbounded acquisition and general continual
learning open. See
`session_records/sequence_working_memory_2026-08-02/repeated_shift_growth_6to8to10to12to14_persistent_v1_2026-08-06/`.

The persistent modular boundary now survives a fifth temporal shift to 62
capabilities. A fresh 32-probe confidence screen remediates only weak
external modules with one additional 256-update block; both seeds pass final
route floors `0.8906` and `0.8594`, hard retention/reversal/recovery,
persistent route/credit reload, corruption rejection, and zero replay. The
no-remediation control remains rejected on the retention gate. This promotes
confidence-triggered external acquisition depth, not unbounded growth,
arbitrary program induction, or general continual learning. See
`session_records/sequence_working_memory_2026-08-02/repeated_shift_growth_6to8to10to12to14to16_remediated_v1_2026-08-06/`.

The same boundary then survives a seventh temporal shift through length 20,
reaching 100 total capabilities. Both replicated seeds pass full-bank
protection, isolated reversal/recovery, persistence, corruption, causal,
shuffled, and zero-replay gates; minimum route floors are `0.8125` and
`0.8594`. This is the current bounded external-growth result, not general
continual learning: the family generator remains finite and arbitrary new
computation, open-ended compression, and positive transfer against a fresh
learner remain unverified. See
`session_records/sequence_working_memory_2026-08-02/repeated_shift_growth_6to8to10to12to14to16to18to20_remediated_v1_2026-08-07/`.

The generated-composition pressure test separately reaches nine sequential
fresh opaque slots plus a tenth target procedure. Both replicated runs pass
strict slot isolation, retention/reversal, reload, corruption, frozen-core,
and zero-replay gates, but the payload remains linear and inherited target
transfer is not positive. This is bounded external capacity growth, not
general continual learning or arbitrary program induction. Evidence is in
`session_records/sequence_working_memory_2026-08-02/generated_composition_opaque_rule_sequential_slot_growth_ten_source_runtime_generated_replicated_promoted_v1_2026-08-07/`.

`ExternalCapabilitySharedResidualBank` is the compact shared-computation
candidate. It keeps one frozen context basis and appends isolated residual
adapters with external recurrent state; old slots and the shared base can be
protected independently. The registry pair passes replicated fresh-outcome
retention and exact reload at `0.5556` of independent-slot payload, while
opaque and third-procedure controls reject. This makes the boundary explicit:
shared computation works for related procedures, but the residual path still
needs additional compute capacity for arbitrary new procedures. See
`session_records/sequence_working_memory_2026-08-02/generated_composition_shared_residual_bank_registry_replicated_boundary_v1_2026-08-07/`.

`ExternalCapabilityResidualComputeBank` is the append-only compute extension:
each new slot gets a compact recurrent context encoder plus an intention
adapter, while the shared context basis and protected slots remain frozen.
It promotes two opaque procedures at two seeds with exact reload and
corruption recovery (`0.8906/0.9023` new-slot behavior, `1.0000` old-slot
behavior). This closes the adapter-only capacity failure but remains bounded;
the next challenge is verified reuse or compression of the local compute
slots. See
`session_records/sequence_working_memory_2026-08-02/generated_composition_shared_residual_compute_opaque_replicated_promoted_v1_2026-08-07/`.

The same residual-compute contract now retains three opaque procedures across
two seeds with reloaded floors `0.8906` and `0.9023` on the middle slot and
`1.0000` on both outer slots. The append-only local compute slots remain
isolated; the next challenge is reuse/compression across them rather than
continued linear growth. See
`session_records/sequence_working_memory_2026-08-02/generated_composition_shared_residual_compute_opaque_three_replicated_promoted_v1_2026-08-07/`.

The residual-compute bank also retains four opaque procedures across two seeds;
reloaded floors are `0.8906` and `0.9453` on the weaker local slots, with all
other slots at `1.0000`. The next challenge is replacing linear local-slot
growth with verified reuse or compression without sacrificing acquisition.
See
`session_records/sequence_working_memory_2026-08-02/generated_composition_shared_residual_compute_opaque_four_replicated_promoted_v1_2026-08-07/`.

`ExternalCapabilityReusableComputeLibrary` now separates physical recurrent
compute from logical binding adapters and states. Related procedures can share
one verified compute module; opaque incompatibility is rejected and must grow
new compute. See
`session_records/sequence_working_memory_2026-08-02/generated_composition_reusable_compute_registry_replicated_promoted_boundary_v1_2026-08-07/`.

`select_reusable_compute_slot` now provides the safe admission rule: reuse a
physical compute module only when all fresh probes clear the floor, otherwise
grow. This enables compression for compatible procedures without blocking
arbitrary new compute. See
`session_records/sequence_working_memory_2026-08-02/generated_composition_reuse_first_grow_policy_replicated_boundary_v1_2026-08-07/`.

`ExternalRegisterBasisCompatibilityPrior` adapts the learned opaque candidate
screen to register-basis signatures. It can learn candidate ordering from
attempted scalar outcomes, but is explicitly screening-only: fresh stable
verifier evidence still decides reuse versus growth. This separates a future
sample-saving prior from the safety-critical admission gate.

The candidate-reuse path now probes every physical compute module, selects the
best fresh-verified binding, and grows only when all candidates fail. This
supports mixed reuse/growth across three opaque procedures while retaining
old states. See
`session_records/sequence_working_memory_2026-08-02/generated_composition_candidate_reuse_opaque_three_replicated_promoted_v1_2026-08-07/`.

The candidate screen now orders those fresh trials from learned-event evidence
and stops at the first independently verified pass. The paired opaque audit
passes two seeds: one ambiguous two-module run saves 19.0% of optimizer
updates and verifier bits with identical behavior and retention, while the
one-module run is correctly neutral. This promotes conditional trial-cost
reduction, not verifier-free admission or general continual learning. See
`session_records/sequence_working_memory_2026-08-02/generated_composition_candidate_screen_opaque_three_replicated_promoted_v1_2026-08-07/`.

The learned page-router growth audit also has a source-only normalization
boundary: the frozen normalizer is fit only on the original source keys while
two independently trained append routers acquire later generations. At the
matched promoted budget, both seeds retain strict candidate/page mastery,
permutation, shuffled-null, immutability, reload, and zero-replay controls.
This is bounded external growth, not unrestricted memory growth or general
continual learning. See
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_page_router_two_generations_66_source_normalizer_promoted_v1_2026-08-08/`.

The same source-only page-router contract survives three independent append
generations: 84 opaque candidates across 30 pages, with replicated strict
mastery, permutation, shuffled-null, immutability, reload, and zero-replay
controls. The retained capability is bounded; routing and verifier work still
grow linearly, so consolidation/compression remains the next frontier. See
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_page_router_three_generations_84_source_normalizer_promoted_v1_2026-08-08/`.

The reusable compute library now separates physical adapter slots from logical
bindings as well as physical recurrent compute. A compatible binding may
share both modules after fresh verifier approval while retaining independent
external recurrent state; no unconditional compression or general continual
learning claim follows.

The external output-candidate store now uses the same stable-address contract
as atomic intention↔entry memory. `ExternalIntentionRepertoire` persists
logical IDs and aliases, maps proposals and composition provenance through
those IDs, and exposes verifier-gated copy-on-write consolidation that
aggregates sufficient statistics without replay. The policy-free runtime can
invoke this maintenance path while the controller remains frozen. This is a
memory-safety and persistence improvement; candidate invention from partial
experience and general continual learning remain open.

`ExternalOutcomeIntentionGenerator` is the next replaceable boundary. It
generates provisional continuous intention content from learned opaque
context, then adapts only external generator state through scalar
score-function credit. It supports delayed feedback, exact Gaussian proposal
log densities, missing-evidence no-ops, protected cells, copy-on-write growth,
and tensor-only reload. Generated vectors are not trusted output: they must
pass the repertoire's held-out verifier-gated admission before factual model
search can treat them as durable candidates. The current five-test causal
rung shows outcome-driven discovery and shuffled-outcome failure, but this is
still bounded proposal generation—not arbitrary program induction or general
continual learning. See `docs/POLICY_FREE_CONTINUAL_LEARNING.md` for the
claim boundary and next experiment.

`ExternalOutcomeIntentionMemory` is the memory-sized successor to that
generator. It lets one controller context query a variable number of external
cells, records planner candidate provenance, and applies delayed score credit
to only the cell whose opaque intention was attempted. The promoted
nonstationary audit in
`session_records/policy_free_intention_memory_2026-08-10/` adds partial-context
masking, noisy reversal, repeated growth, and transactional rejection of
negative transfer. `ExternalOutcomeIntentionRouter` now adds a replaceable
memory-side context-to-cell route policy. The runtime receives only the
selected opaque intention; route propensity and delayed scalar feedback remain
in the external state, and append/protect/persistence semantics are preserved.
Routing occurs before content generation, so sparse proposals materialize only
the selected physical cell IDs instead of the whole external bank for a
single-context step.
The replicated audit in
`session_records/policy_free_intention_routing_2026-08-10/` promotes caller-free
bounded routing with shuffled, missing-evidence, corruption, frozen-core, and
zero-replay controls. It does not claim unrestricted growth, compression,
arbitrary new computation, or general continual learning. A matched fresh-cell
control now shows positive successor transfer, while route-cost scaling,
compression, and stable-prefix operation beyond this bounded audit remain
open.

`PolicyFreeAmodalRuntime` is the canonical integration seam for that generator.
It proposes from the controller's opaque adapted state, leaves generator state
mutation to explicit caller-side feedback methods, and can combine a
provisional generated candidate with an independently verified repertoire
without changing the controller. The two-seed audit in
`session_records/policy_free_intention_generation_2026-08-10/` passes frozen
core, copy-on-write retention, fresh-transfer, shuffled-outcome, persistence,
and zero-replay gates. This promotes the seam, not unrestricted continual
learning.

`ExternalProgramAmodalRuntime` is the canonical execution seam for the
CPU-plus-files path. It runs the frozen amodal controller, a replaceable
`ExternalCapabilityRegisterMachine`, and the intention bus in one
`INPUT -> PROCESS -> OUTPUT` cycle. Portable `ExternalProgramArtifact` files
are observed and executed copy-on-write outside the controller; only the
resulting opaque intention reaches decoders. Use verifier-gated external
retention when promoting a new file. This provides a stable path for learning
new computation without adding controller branches, but does not by itself
claim learned program synthesis or general continual learning.

`ExternalRegisterComputeBasisArtifact` closes the corresponding persistence
gap for learned external computation. A compute slot's ABI and tensor state
can be checksummed, moved between interpreters, and reloaded through
`ExternalCapabilityRegisterMachine.add_basis_artifact()` without mutating the
shared interpreter. Instruction vectors and compute capacity are now both
portable external files; retention and behavior verification still decide
whether a loaded slot is deployable.

Compute bases also support `register_input_mode="event_window_only"`. This
mode makes a newly appended file read its own persistent standardized event
window and opaque instruction without depending on a prior file's hidden
register distribution. It is the generic isolation boundary used by the
rendered triplet-parity growth audit; it does not assign semantic meaning to
register coordinates or add a task-specific branch.

The external intention learner also supports an opt-in `context_masking=True`
mode. Callers pass `context_mask=` alongside an opaque context; the memory
learner receives explicit observed-value and observation-mask channels, so
missing dimensions are not silently learned as zeros. Routing credit uses only
observed values, and retention prototypes keep per-dimension observation mass.
Dense mode remains backward-compatible. Routed-memory v1 payloads migrate with
empty observation history, while v2--v3 dense payloads preserve their old
prototype behavior. This makes partial evidence safer; it is not yet general
missing-stream cognition or general continual learning.

The canonical policy-free runtime forwards this contract through
`observe(..., intention_context_mask=...)`, so the controller remains frozen
while the replaceable memory side receives the evidence mask.

In masked router mode, route keys receive the same value/mask representation.
Copy-on-write children keep reusable value weights but neutralize
mask-specific weights and dimensions never observed by the source cell. Verified
cells also carry a generic support prior: a protected route is downweighted when
its verified prototype covers too little of the query's observed evidence. In
masked mode, repeated relevant low outcomes quarantine a protected cell rather
than mutating it, so a copy-on-write challenger can learn a reversal without
forgetting the old file.

The overlapping-mask audit now promotes this boundary across two seeds; see
`session_records/policy_free_intention_masked_routing_overlap_promoted_2026-08-10/`.
The halfway-switch gradual curriculum is archived as a rejection in
`session_records/policy_free_intention_masked_routing_gradual_rejected_2026-08-10/`.
The seven-stage mask-drift curriculum is also rejected in
`session_records/policy_free_intention_masked_routing_multistage_rejected_2026-08-10/`:
the single cell does not remain stable when evidence distributions change
sequentially. The next architectural step is versioned or factored reusable
computation across those distributions.

The router now also supports an opt-in context-versioned memory boundary. Each
masked cell persists an opaque observation-mask profile, and a fork can freeze
the superseded cell, copy its reusable content, and learn a fresh route address
for the new evidence distribution. Routed-memory v5 persists this profile and
migrates the prior v4 support state. This is a safer foundation for continual
learning, but its versioned multi-stage audit is rejected for promotion: both
seeds preserve retention and persistence, yet neither produces a replicated
warm-over-fresh speedup. See
`session_records/policy_free_intention_masked_routing_versioned_rejected_2026-08-10/`.
Arbitrary missing-stream reasoning, unrestricted growth, compression, and
general continual learning remain unqualified.

The generator also exposes an opt-in factorized masked-content ABI. With
`mask_stable_content=True`, the observation mask remains available to routing
and retention but cannot mutate the nonlinear hidden content program.
`factorized_context_residual=True` adds a separate learned residual from
observed values and bias to the opaque intention. This keeps evidence-specific
adaptation in independently replaceable external state, supports delayed
score-function credit, copy-on-write growth, protected retention, and exact
schema migration, including zero-initialized upgrades from compatible legacy
generator files, without adding a controller branch. The bounded
overlapping-mask audit promotes this mechanism across two seeds; the
multi-stage versioned curriculum remains rejected. See
`session_records/policy_free_intention_masked_routing_factorized_promoted_2026-08-10/`.

The external routing harness also supports
`adaptive_versioned_multi_stage`. Instead of forcing fixed update boundaries,
each evidence version must pass stage-local mastery and a held-out prefix
verifier before a protected child is created. Adaptive forks copy the route
key on dimensions already observed and temporarily raise the unqualified-cell
exploration floor to `0.75`, preserving caller-free discovery of the new file.
The three-seed audit promotes bounded seven-stage sequential reuse with
warm/fresh update counts `39/50`, `42/44`, and `34/55`; arbitrary distribution
shift and general continual learning remain unqualified. See
`session_records/policy_free_intention_masked_routing_adaptive_promoted_2026-08-10/`.

The router also supports verifier-selected copy-or-fresh admission through
`select_verified_transfer_prior`. It probes isolated inherited and fresh
external cells with outcome-only evidence, verifies that the live source was
not mutated, and returns a versioned selection receipt plus the chosen state.
This prevents negative transfer from being deployed blindly while preserving
the frozen controller and replaceable memory boundary. The three-seed novel
challenger audit is archived in
`session_records/policy_free_intention_novel_challenger_promoted_2026-08-10/`.
It promotes bounded admission safety, not positive novel-task transfer or
general continual learning.

The companion positive-transfer harness uses the same API with an unseen
nearby target. Transfer wins all six isolated branch probes across three
seeds, while the negative-transfer harness continues to select fresh state.
Both records are retained under `session_records/` and deliberately keep
universal speedup and broad generalization unclaimed. Selection receipts also
support v2 cost-aware adjusted scores for deployment-budget decisions.

The sequential admission harness extends this to three unseen families with
append-only growth and complete-prefix retention after every admission. Its
three-seed record is archived at
`session_records/policy_free_intention_sequential_admission_promoted_2026-08-10/`.

The sequential harness now retrieves its copy source automatically from
verified external cells with `select_verified_source_cell`; no physical cell
index is passed by the caller. Source-selection receipts preserve the
learned compatibility and coverage evidence for audit and persistence. The
three-seed record is archived at
`session_records/policy_free_intention_sequential_auto_source_promoted_2026-08-10/`.

`ExternalRoutedIntentionCostModel` provides the complementary memory-side
admission economics. Masked opaque context, verified source coverage, and
bank size predict transfer/fresh continuation cost; only the selected branch
learns from its normalized completed cost. The controller remains unaware of
both the physical source and the cost policy. The three-seed audit is
archived at
`session_records/policy_free_intention_learned_cost_promoted_2026-08-10/`.

`ExternalRoutedIntentionCostLedger` is the stateful form used by online factual
transition routing. It shares one versioned cost policy across stream-local
routers, updates only after a candidate passes held-out and retention
verification, and persists independently from the controller and factual model
bank. This is an external learning-memory seam; it is not yet evidence that
cost prediction generalizes across broad task families.

Factual transition-bank prior selection also accepts optional opaque transfer
and fresh acquisition costs. Its v2 receipt records raw and cost-adjusted
probe errors, while zero costs retain the v1 error-only behavior. The choice is
copy-on-write and reversible, so a costly warm prior cannot silently displace a
cheaper fresh challenger or mutate a mastered source.

`ExternalSequenceProgramMemory.admit_verified_artifact()` is the matching
memory-side file transaction. It stages an ABI-checked artifact, consumes only
ordered scalar verifier outcomes, and appends it only after a stable prefix
passes. Rejection is non-mutating; protected files remain untouched; and
`payload()` / `from_payload()` persist the opaque artifacts, routing tensors,
output ABI metadata, and protection state independently of the controller.
This makes executable files safe external learning state without claiming that
the system can synthesize arbitrary programs yet.

`ExternalProgramCandidateSearch` adds the first outcome-driven synthesis seam
for that file format. It mutates opaque instruction sequences outside the
controller, records only aggregate verifier statistics and candidate digests,
and leaves durable admission to the existing stable-prefix transaction. The
promoted three-seed audit synthesizes one held-out two-step composition from a
protected parent while a matched fresh atom fails the same bounded search
budget. This qualifies one-edit structural synthesis only; multi-step beam
search, arbitrary program induction, and general continual learning remain
open.

## Persistent multi-step external hypotheses

`ExternalProgramHypothesisFrontier` is the memory-side continuation of
`ExternalProgramCandidateSearch`. It performs bounded generic composition of
opaque executable files while keeping the controller, interpreter, and
protected capability files unchanged. Provisional hypotheses include only
opaque artifact tensors, parent checksums, depth, scalar quality, and
replay-free aggregate search state. `payload()` / `from_payload()` preserve
the frontier exactly without serializing raw verifier outcomes.

The default frontier mode is exhaustive breadth-first local expansion from a
replaceable instruction bank. It is a correctness-oriented finite mode for
multi-step composition; `proposal_mode="stochastic"` retains the learned
operator-prior path for future scale experiments. Admission remains an
independent stable-prefix transaction, so failed hypotheses cannot overwrite
protected external files. This establishes a bounded CPU-plus-files search
seam, not arbitrary program induction, unrestricted growth, or general
continual learning.

## Verifier-gated executable-memory lifecycle

`ExternalSequenceProgramMemory` now behaves like a versioned external file
store rather than an append-only tensor list. Stable logical IDs survive
physical removal. `evict_verified()` and `consolidate_verified()` build
copy-on-write candidates and commit only after caller-owned held-out retention
and equivalence probes pass. `compressed_payload()` and
`compress_verified()` provide checksummed durable compression with a
post-decompression behavior gate; the hot controller boundary remains
unchanged.

The three-seed promotion is archived at
`session_records/sequence_working_memory_2026-08-02/external_program_memory_lifecycle_promoted_2026-08-10/`.
It demonstrates safe bounded lifecycle management with a frozen controller,
zero replay, and zero controller updates. It does not claim learned
maintenance selection, learned compression, unrestricted growth, arbitrary
new computation, or general continual learning.

## Learned executable-memory maintenance

`ExternalSequenceProgramMemory.maintenance_features()`,
`propose_maintenance()`, and `apply_maintenance_proposal()` connect the
generic memory-side maintenance learner to real executable-file transactions.
The adapter exposes only generic storage telemetry and a structural action
mask. `grow`, `share`, `compress`, and `evict` still require the independent
verifier-gated admission, equivalence, retention, or checksum paths; `defer`
is a non-mutating no-op. The controller and interpreter remain unaware of
logical file IDs and maintenance decisions.

The promoted audit is archived at
`session_records/sequence_working_memory_2026-08-02/external_program_memory_maintenance_promoted_2026-08-10/`.
It qualifies replay-free learned lifecycle choice on a bounded executable
workload, not general continual learning or unrestricted program acquisition.

## File-scoped executable working state

`ExternalProgramAmodalRuntime` keeps a separate recurrent
`ExternalRegisterState` for every stable logical file ID in
`ExternalSequenceProgramMemory`. Routing between files no longer carries one
file's temporal context into another. State for newly admitted files is
created lazily, and state for verified-retired IDs is pruned on the next
runtime step. The controller remains fixed-size and unaware of file identity.

The runtime schema is `neural-computer.external-program-runtime.v6`. Version 6
uses the replaceable trajectory query by default, so executable-memory routing
sees the post-step controller representation plus masked event-window
statistics. Different rows may route to different files in one tick, with
row-partitioned execution snapshots and no cross-file state writes. The
final-state adapter remains available as an explicit compatibility choice.
The output additionally carries the opaque route query and soft file
probabilities for host-side delayed scalar credit and exact propensity
accounting; neither is returned to the controller.
Route exploration is disabled by default. When enabled with
`program_route_exploration`, the output also reports the exact selected-file
propensity for delayed scalar credit.

Pass an `ExternalOutcomeProgramRouter` to make route learning part of the
runtime state machine. The previous scalar outcome updates only that external
router before the next file selection; `activate_program()` appends one newly
admitted file to the route capacity without resizing the controller. Router
eligibility and policy state are included in the checksummed runtime
checkpoint. Eviction or compaction without an explicit route-policy migration
is rejected at execution time rather than silently changing file meaning.
`step_events(..., route_feedback=...)` can optionally deliver delayed scalar
credit to the external router while keeping the controller's own feedback
stream separate; omitting it preserves the historical shared-feedback
behavior.

For a larger bank, `ControlFlowProgramAmodalRuntime` can attach a replaceable
`ExternalControllerTrajectoryQueryAdapter`. Its detached query combines the
opaque post-step controller representation with bounded event-window
statistics and is exposed only to the external router. The promoted
four-file counterfactual audit sends full candidate outcome vectors to that
router, reaches `1.0000` held-out accuracy across four seeds and reversed file
orders, and keeps the controller frozen with zero replay. This remains a
bounded route-bank result, not arbitrary new computation or general
continual learning; evidence is archived at
`session_records/control_flow_runtime_four_file_counterfactual_promoted_2026-08-12/`.

`ExternalProgramRuntimeState.payload()` and `from_payload()` provide a
versioned tensor-only pause/resume checkpoint for the controller working state
and every isolated executable-file state. Executable artifacts and model
parameters remain separate resources; the checkpoint stores no raw modality,
protocol, or verifier-private data, and its envelope checksum rejects silent
tensor corruption.

`ExternalProgramFastCell` is an optional memory-side extension for the same
runtime. Each logical executable file can own an isolated outcome-gated
fast-weight cell whose read is exposed as protected-meta execution context.
The cell query uses only the learned controller representation and opaque
intention; a positive opaque action/outcome record writes its external state.
Failed or missing feedback is an exact no-op. The runtime persists the cell
states and the previous logical-file/query binding, so delayed feedback cannot
be credited to a newly selected file. The cell is zero-effect at construction
and does not resize or update the controller. Use `state_payload()` and
`state_from_payload()` on the runtime when this optional cell is configured.

The bounded transfer audit for this seam passed on seeds `69316` and `69317`:
the frozen source codec reached stable target behavior on the first newly
allocated file, while matched fresh codecs required 130 and 116 stable target
lifetimes. This validates reusable memory-side computation, not arbitrary new
procedure acquisition or general continual learning. The next required gate
is a rendered Brain Workshop transfer audit with complete-prefix retention and
zero-replay controls. See
`session_records/external_program_fast_cell_transfer_2026-08-11/`.

The canonical runtime now also accepts `ExternalWorkingMemoryCell` as a
versioned causal memory boundary. It reads old state before appending the
current learned event/action/outcome row, supports tensor-only persistence and
capacity growth, and is independent of controller weights. The replicated
causal audit reaches `1.0` fresh-state n-back-2 accuracy on two seeds with a
frozen codec; n-back-3 remains at chance. See
`session_records/brainworkshop_causal_working_memory_transfer_2026-08-11/`.

The policy-free factual seam also exposes
`ExternalControllerEventWindowStateAdapter`. It preserves the compact opaque
controller state while folding bounded event-window statistics into the same
planner width, so short-evidence transition learning can retain more temporal
binding without enlarging the external model. Promotion can require a real
recursive held-out rollout in addition to one-step prediction, source-slot
retention, and a matched fresh challenger. This remains a bounded frozen-core
transition boundary; multi-lifetime promotion and general continual learning
are still open.

Transition-slot promotion supports multiple independent held-out observations
and rollouts. Selection and recursive verification use the worst held-out
error, so a candidate cannot enter external memory after succeeding on only a
single lucky lifetime. The rendered Brain Workshop audit uses three staged
holdouts plus a post-promotion route lifetime; its seed-mixed result remains a
bounded safety boundary rather than a claim of general continual learning.

The online transition router also supports an explicitly external provisional
context-continuity route. A shifted candidate stream can remain attached to its
isolated staged slot when opaque context similarity and a bounded factual error
agree, even if strict one-step continuation has not stabilized. This route is
read-only with respect to committed memory and is margin/threshold gated. The
replay-free audit uses a mixed bank that selects between affine and
random-feature sufficient-statistics candidates; its accounting reports a
mixed bank as replay-free only when all committed slots are actually
replay-free.

The frozen-core event-window adapter has two external statistics contracts:
the compatibility `masked_mean_and_max_v1` summary and the causal
`recency_weighted_and_latest_v1` summary. The latter preserves more ordering
information without resizing the controller or planner state. It is selected
by `window_statistics`, `window_gain`, and `recency_decay`; these are memory
boundary configuration, not modality- or task-specific reasoning branches.

Discovery also exposes a committed-slot write firewall: callers can consume
rows into isolated provisional candidates while suppressing adaptation of
temporarily matched committed slots. Promotion is the only path that commits
the new external model. This keeps source retention independent of transient
route ambiguity.

## Opaque goal-fragment memory

`ExternalGoalFragmentMemory` stores destinations outside the fixed controller.
Each `ExternalGoalFragmentSet` is a runtime-sized collection of opaque target
vectors and learned/verified masks. The factual planner can compose them as a
`union` (any fragment) or `intersection` (all fragments), so behavior is still
derived by model-based search rather than retrieved as a task policy. The
policy-free runtime can read fragments by external indices; addresses and
composition metadata never enter the controller. Admission is copy-on-write,
checksum-protected, and independently reloadable.

## Shared-basis content-addressed memory

`SharedBasisContentAddressedMemory` and its persistent subclass factorize only
stored value payloads. Independent opaque keys and logical records remain
stable while coefficients reference an appendable external orthonormal basis.
The ordinary `MemoryRead` and `MemoryCandidates` interfaces materialize values,
so the controller remains storage-agnostic. `compression_candidate()` creates
a copy-on-write representation, and `replace_from_candidate()` requires an
optional retention verifier plus an expected store version before committing.

The canonical two-seed pressure test is archived at
`session_records/brainworkshop_external_temporal_shared_basis_compression_promoted_2026-08-12/`.
It qualifies verifier-gated shared storage compression only; rank selection is
deterministic in this first boundary audit.

`OpaqueSharedBasisCompressionPolicy` is the external learned selector for
runtime-sized shared-basis candidates. It consumes only generic candidate
statistics and learns from one scalar verifier utility per proposal. Its
candidate index is advisory; `SharedBasisContentAddressedMemory` still owns
route/value verification, expected-version checks, and atomic persistence.
The canonical policy-growth audit is archived at
`session_records/brainworkshop_external_temporal_shared_basis_policy_growth_promoted_2026-08-12/`.

`OpaqueSharedBasisStructurePolicy` is the stricter raw-value variant. It reads
opaque value rows and occupancy, computes a fixed-width
row-permutation-invariant singular-spectrum summary, and learns rank choice
from scalar verifier utility without receiving precomputed candidate
reconstruction error. It remains external to the controller and emits only a
candidate index; `SharedBasisContentAddressedMemory` still owns independent
route/value verification, versioned copy-on-write, and persistence. The
canonical two-seed transfer audit is archived at
`session_records/brainworkshop_external_temporal_shared_basis_structure_growth_promoted_2026-08-12/`.

The v2 structure policy also has a repeated-growth audit: its spectral plus
normalized pairwise summary transfers rank selection through three successive
external memory growth transitions while ignoring unoccupied padding. The
policy and controller remain external/frozen respectively; the shared-basis
memory independently verifies each copy-on-write replacement. Evidence is
archived at
`session_records/brainworkshop_external_temporal_shared_basis_repeated_growth_promoted_2026-08-12/`.

The competing-subspace audit extends the external structure policy to runtime
ranks `(2, 4, 8)` and verifies dynamic shared-basis growth across incompatible
subspaces. It preserves the same storage-agnostic controller boundary and
independent verifier-gated persistence. Evidence is archived at
`session_records/brainworkshop_external_temporal_shared_basis_competing_subspaces_promoted_2026-08-12/`.

`SharedBasisContentAddressedMemory` also exposes `rewrite_candidate()` and
`replace_from_rewrite_candidate()` for verifier-gated logical row replacement
inside one scope. Other scopes remain isolated and unchanged; the controller
still sees only ordinary materialized memory reads. The protected-scope regime
replacement audit is archived at
`session_records/brainworkshop_external_temporal_shared_basis_regime_replacement_promoted_2026-08-12/`.

`OpaqueRegimeChangePolicy` is the external learned trigger for that rewrite
boundary. It compares opaque current and incoming value banks using
permutation-invariant spectral and cross-bank features, adapts from scalar
verifier utility, and emits only a keep/replace plan. The stable path is an
exact byte/version no-op; the shifted path remains subject to the memory
verifier, expected-version check, and protected-scope isolation. It does not
add a modality-specific reasoning branch or enter the frozen controller. The
canonical two-seed audit is archived at
`session_records/brainworkshop_external_temporal_shared_basis_learned_regime_trigger_promoted_2026-08-12/`.

The alternating-regime audit drives the same external trigger through five
hidden working-scope reversals while three protected scopes remain isolated.
Each accepted rewrite replaces the working logical rows and then reuses the
shared-basis representation; no stale occurrence is appended. Stable probes
are exact memory no-ops, and the controller/encoder remain frozen. Evidence
is archived at
`session_records/brainworkshop_external_temporal_shared_basis_alternating_regimes_promoted_2026-08-12/`.

`GatedResidualRegimeChangePolicy` is the first explicit parameter-isolated
anti-forgetting boundary for online external adaptation. It keeps an
immutable `OpaqueRegimeChangePolicy` as the fallback and grows a zero-start
residual from scalar verifier utilities. The residual can override only when
its evidence is positive and stronger than the base action, preserving the
old capability while new external state learns. Its canonical partial-overlap
audit, including the rejected unconstrained-update control, is archived at
`session_records/brainworkshop_external_temporal_regime_policy_online_adaptation_promoted_2026-08-12/`.

`GatedResidualRegimePolicyBank` adds isolated residual slots behind an opaque
binding-context key. It preserves the frozen base detector as fallback,
routes contexts by an external cosine key, and exposes per-slot trainable
parameters so later growth cannot mutate an earlier slot. This is the binding
boundary needed before attempting general multi-capability continual
learning; it does not assign semantic meaning to keys. Evidence is archived
at
`session_records/brainworkshop_external_temporal_regime_policy_binding_slots_promoted_2026-08-12/`.

Slots can be verifier-promoted and frozen through `freeze_slot()`, after which
their trainable-parameter path rejects updates. An optional `max_slots` bound
rejects unverified capacity growth. `slot_replacement_candidate()` and
`replace_slot_from_candidate()` provide copy-on-write, verifier-gated reuse of
a full slot without mutating live state on rejection.

The generic `ExternalCapabilityEvictionPolicy` can rank the bank's opaque
candidate summaries before this transaction. It is trained outside the
controller from scalar verifier utility and does not receive physical slot
indices or semantic names. The learned maintenance audit is archived at
`session_records/brainworkshop_external_temporal_regime_policy_learned_maintenance_promoted_2026-08-12/`.

`GatedResidualCapabilityEvictionPolicyBank` extends the same lifecycle to
nonstationary maintenance objectives: an immutable base scorer handles unknown
contexts, while independently activated/frozen residual scorers learn distinct
candidate rankings from fresh scalar utilities behind opaque binding keys.
Evidence is archived at
`session_records/brainworkshop_external_temporal_regime_policy_nonstationary_maintenance_promoted_2026-08-12/`.

`EpisodicBindingRouter` removes the remaining requirement that an experiment
hand in a context key. It encodes learned event/action/outcome trajectories,
provisions opaque keys from observed contexts, and updates only its external
encoder from the utility of the slot that was actually attempted. After
promotion, the encoder can be frozen while the opaque key bank remains
replaceable external state. The two-seed learned-binding audit reached perfect
forward and candidate-order-permuted routing, retained behavior after exact
reload, and stayed at chance under reward shuffling. Evidence is archived at
`session_records/brainworkshop_external_temporal_learned_binding_routing_promoted_2026-08-12/`.

The v3 binding router adds an immutable generic episode signature alongside the
learned route embedding. Signature-keyed novelty evidence is kept separate
from the trainable route score, and key consolidation plus replacement are
copy-on-write and retention-probed. This allows a frozen router to recognize
and safely admit a novel external binding without silently assigning it a
retired slot. The online capacity audit is archived at
`session_records/brainworkshop_external_temporal_online_binding_capacity_promoted_2026-08-12/`.

The generic `ExternalCapabilityEvictionPolicy` can now drive the router’s
capacity transaction: it ranks opaque binding candidates from incoming
signature plus generic reliability/age telemetry, while the verifier retains
protected siblings and authorizes copy-on-write replacement. The promoted
victim-selection audit demonstrates transfer under candidate permutation with
zero controller updates; the policy remains an external replaceable learner,
not a controller reasoning branch.

`EpisodicBindingArchive` is the long-term memory-side file boundary for the
episodic router. It appends immutable learned context/signature records and
generic scalar reliability/age telemetry while keeping only a bounded active
slot residency map. Eviction clears cache residency rather than deleting the
record; a later signature lookup can reactivate the record without replaying
the old training stream. It has an independent versioned payload and does not
add a controller or modality-specific reasoning branch.

The repeated interleaved archive audit is archived at
`session_records/brainworkshop_external_temporal_interleaved_binding_archive_promoted_2026-08-12/`.

`EpisodicBindingArchive` v2 adds a cached normalized signature matrix and
`lookup_many()` for batch retrieval, explicit protection latches and reversal
streak/count telemetry, and a canonical SHA-256 checksum over serialized
payloads. v1 payloads remain readable for migration, while new payloads are
integrity-checked. `snapshot()` / `load_snapshot()` provide a compact tensor
snapshot for durable storage; the scale audit reduced a 1,024-record archive
from about 645 KB JSON to about 166 KB while preserving retrieval. The
scale/reversal audit is archived at
`session_records/brainworkshop_external_temporal_archive_scale_reversal_promoted_2026-08-12/`.

## Canonical external temporal-history bridge

`ExternalTemporalHistoryEventBridge` and
`AmodalControllerRuntime.step_streams_with_external_history()` now connect the
replaceable temporal store to the production `INPUT -> PROCESS -> OUTPUT`
boundary. On each tick the bridge reads caller-selected relative offsets
before appending the current learned event tokens. Current and historical
tokens remain separate on the event axis, with history preceding the current
token so the controller's latest-event semantics remain correct. Historical
tokens are transient processing context; only current tokens enter the
controller's persistent event window. Missing records remain explicit
`present=False` tokens rather than fabricated evidence. The runtime rejects a
query that exceeds the controller's bounded processing window before mutating
external memory.

This is an integration and causality contract, not a general continual-
learning result. The v1 store persists learned payloads and derives historical
confidence from presence; it fails closed when source keys, timestamps, or
durations are present. The explicit v2 store preserves those fields and their
per-token presence masks, with a fixed source-key width selected at construction
time. Offset selection is still caller/trainer state, and the controller remains
bounded by its event window. Learned query-conditioned addressing, history
compression, and unrestricted continual learning require separate promotion
experiments.

`ExternalTemporalAddressIndex` is now the canonical content-addressed extension
for this boundary. It stores only a learned opaque key, an index namespace, and
an opaque `(target_scope, target_position)` location; event payloads remain in
the target store. For `ExternalTemporalHistoryMemory`, target positions are
absolute within a scope, so later appends do not silently retarget an older
record. A lookup miss returns `hit=False`, `-1` location fields, and an absent
history token. The runtime path
`AmodalControllerRuntime.step_streams_with_external_address()` passes the
already-resolved read directly to the bridge, preserving the stable address
without converting it into a shifting relative offset. This remains an
external storage contract, not a learned capability claim.

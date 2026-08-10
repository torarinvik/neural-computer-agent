# Policy-free continual learning architecture

This is the architectural consequence of the latest continual-learning
pressure tests and the exported games-session findings. The objective is not
to preserve an arbitrary policy while it is continually rewritten. It is to
learn reusable factual structure once, then derive behavior for each new goal
at inference time:

> After learning A, a novel B should require fewer verified experiences than a
> fresh learner, while previously acquired capabilities remain intact.

## Core distinction

A policy is preferential: it says what to do. A new task can contradict that
preference, so fine-tuning it creates negative transfer and catastrophic
forgetting. A transition model is factual: it predicts what an opaque state
becomes after an opaque intention. New observations add or refine facts; they
do not need to overwrite a preference surface.

The canonical flow is therefore:

```text
encoded events -> one frozen amodal controller -> opaque state/goal
                                      |
                                      v
             external transition model + persistent model memory
                                      |
                                      v
                         model-based intention search
                                      |
                                      v
                         intention bus -> decoders
```

The controller remains the sole general cognitive model. The model-side
components are replaceable memory and computation infrastructure, not a second
task-specific reasoner and not a protocol decoder.

## Three levels of durable knowledge

Experience should be stored at the level at which it generalizes:

1. **Shared grammar in stable parameters.** Train from structurally diverse
   source regimes so domain-specific explanations disagree and reusable
   transition structure is the part that survives. Diversity is a
   generalization instrument, not merely more data.
2. **External regime/model slots.** Store factual dynamics that apply to a
   particular opaque context or local universe. Slots are independently
   versioned, retrievable, and evaluated without changing the controller.
3. **Verified delta/exception records.** Represent a new regime as a small
   difference from the nearest existing model when possible. Commit it
   copy-on-write only after a held-out outcome probe verifies both the delta's
   new capability and the complete retained floor.

This is the neural equivalent of replacing one ill-fitting puzzle piece, not
repainting the entire puzzle. A delta may be a residual model, a sparse route
record, or a signed entry whose polarity changes an existing prediction. The
representation is learned and opaque; “delta” describes storage behavior, not
hand-authored semantics.

## What must not be canonical

The following are useful experiments but are not the long-term continual
learner:

- a learned action or eviction policy that must be fine-tuned for every new
  regime;
- freezing a policy while moving all plasticity into an adapter;
- a replay buffer that silently becomes the real learner;
- a task-specific branch, rule label, or privileged target-action field;
- a capacity decision trained on structurally impossible actions.

The current opaque capacity planner and its residual adapter remain valid as
auditable memory-side experiments. Transaction legality must be supplied by
the caller; learned scores may rank only the legal candidates. General
behavior should come from factual model search, not from this policy.

## Required learning protocol

Every promoted experiment must separate:

- zero-shot capability from target adaptation;
- target updates from cumulative lifetime cost;
- model learning from inference-search expansions and latency;
- fresh-target learning from transfer;
- current-target mastery from retention of every prior regime;
- unique verifier bits from optimizer updates and replayed examples.

The minimum controls are a fresh learner, a passive or reward-shuffled learner,
valid pixel/rerender changes where applicable, model corruption, source or
goal shuffles, and byte-stability checks for retained model slots. A transfer
claim is not valid when the target is so easy that a fresh learner has no
headroom, or when warm-up cost is charged to only one of many later targets.
When a transfer challenger is used, its fresh candidate and the matched fresh
control must begin from the same caller-owned initial state. Otherwise random
initialization is confounded with transfer and the cumulative cost comparison
is not interpretable.

## Repository mapping

The factual/search boundary is implemented by
`neural_computer.ExternalTransitionModel`,
`neural_computer.ExternalTransitionModelBank`, and
`neural_computer.ExternalModelBasedPlanner`. The amodal controller and
intention bus remain in `src/neural_computer/`; the model and planner remain
replaceable external components. The sequential audits under
`experiments/external_transition_model_compounding/` and
`experiments/external_transition_model_disjoint_compounding/` are the current
promotion path for this architecture.

The interleaved bounded-memory experiment is a complementary pressure test for
binding, delayed evidence, eviction, and retention. Its structural action mask
is part of the transaction boundary; its learned capacity policy is not
evidence of general continual learning.

The shared-bank stream boundary is implemented by
`neural_computer.ExternalMultiStreamTransitionContextRouter`. It is the
policy-free analogue of the exported games session's shared-controller design:
one controller and one factual bank serve many opaque streams, while each
stream owns only its transport window and provisional copy-on-write state.
Promoted stream addresses are preferred continuations, not semantic labels, and
every new factual write still requires independent held-out verification plus
retention. The current pressure test is bounded and synthetic; it is an
architecture invariant, not evidence that the system has learned arbitrary
identity formation.

The follow-on robustness audit in
`experiments/external_transition_model_multistream_robustness/` adds the
bounded lifecycle controls: a missing stream keeps only its own pending row,
contradictory evidence becomes a non-mutating conflict, and a drifted stream
is replaced only after verifier-gated eviction and retention of the sibling
stream. This is still caller-owned opaque binding; the remaining generality
bottleneck is learned identity/delay/reliability formation from asynchronous
events rather than externally supplied stream keys.

## Learned anonymous binding boundary (2026-08-10)

The next layer is implemented by `ExternalOnlineStreamBindingMemory` and
`ExternalLearnedMultiStreamTransitionContextRouter`. A frozen
`ExternalTransitionContextEncoder` is trained from paired same-stream views;
deployment then grows only external state: anonymous track keys, bounded
transition prefixes, opaque prototypes, inter-arrival delay estimates, and
positive/negative verifier sufficient statistics. The combined router feeds
those learned keys into the same shared factual bank, so the caller no longer
supplies stream identity.

The two-seed pressure test in
`experiments/external_learned_stream_binding/` separates three anonymous
streams with 100% diagnostic consistency versus 16.7% for a fresh untrained
encoder. It passes missing-arrival isolation, interleaving-order permutation,
learned delay, verifier reliability, frozen-controller, exact-persistence,
and checksum-rejection gates. Trainer-only stream indices construct positive
pairs and score the diagnostic; they never enter deployed memory.

This qualifies a bounded learned identity/binding boundary, not general
continual learning. The next pressure must vary encoders, stream counts, delay
laws, open-set arrivals, and contradictory evidence while retaining the
factual router's held-out promotion and complete-retention gates.

## Open-set learned binding with transactional replacement (2026-08-10)

The learned binding boundary now distinguishes evidence collection from
identity admission. `ExternalOnlineStreamBindingMemory` keeps a bounded set of
anonymous provisional tracks when live capacity is exhausted. Provisional
tracks can collect observations, delay statistics, and verifier outcomes, but
they cannot emit a live stream key or mutate the shared factual model bank.

Promotion and retirement use copy-on-write state plus a caller-owned retention
probe. A failed probe is a no-op over the entire binding state, not merely over
one prototype. This is the memory-side analogue of retaining every mastered
primitive while learning a new one: replace only after the candidate proves
that the retained floor is intact. The result is a versioned, checksummed
boundary suitable for a growing external memory, while the controller remains
frozen and protocol-agnostic.

The two-seed open-set audit (`2301`, `2302`) used four anonymous streams, three
live slots, six irregularly timed arrivals for the fourth stream, zero replay,
and zero controller updates. Both seeds quarantined the unseen stream,
rejected unsafe admission and retirement without mutation, then admitted it
after verified retirement and exact persistence. This promotes bounded
open-set lifecycle safety only. It does not claim that the system has learned
when to evict, discovered arbitrary identities, handled unrestricted drift, or
achieved general continual learning. The next step is an outcome-trained
retention/admission challenger with simultaneous provisional identities and
adversarial contradiction controls.

## Outcome-trained lifecycle proposals (2026-08-10)

The anonymous binding memory now separates three concerns cleanly: provisional
evidence collection, learned proposal ranking, and verifier-authorized commit.
`ExternalStreamBindingLifecyclePolicy` is a replaceable external learner. It
receives opaque prototype vectors plus generic lifecycle telemetry, records its
selection propensity, and updates from one scalar verifier outcome without
replaying the source evidence. It can learn to hold when every provisional
candidate is contradictory.

The commit path is atomic: one provisional identity and one live identity are
swapped on a copy, then the complete retained state is verified before commit.
Rejection leaves the original state unchanged. This is materially closer to a
growing file-like memory behind a fixed compute substrate than a policy adapter
inside the controller; the controller never receives the lifecycle decision or
raw modality data.

The two-seed lifecycle audit (`2401`, `2402`) used five anonymous streams,
three simultaneous provisional identities, fresh and outcome-shuffled controls,
contradiction/hold evaluation, exact policy persistence, and zero replay. The
trained policy reached `1.0` safe-replacement and `1.0` contradiction/hold
accuracy on both seeds, while fresh controls reached `0.125`/`0.1667` and
shuffled controls `0.125`/`0.2083`. The result promotes outcome-trained
proposal ranking and atomic retention safety only. Learned verifier design,
autonomous eviction policy, unrestricted growth, and general continual
learning remain open.

## Joint learned binding and factual retention (2026-08-10)

The binding policy is now coupled to the factual memory boundary through
`ExternalLearnedMultiStreamTransitionContextRouter.replace_with_factual_candidate`.
The policy still proposes only an anonymous provisional/live pair. The external
memory layer performs the consequential work on copy-on-write binding and
multi-stream factual-router state: it evicts the retired factual slot, consumes
the provisional evidence once with streaming affine sufficient statistics,
checks an independent held-out transition, and commits both replacements only
after the scalar verifier outcome authorizes the transaction. A scalar rejection
or a wrong held-out transition leaves both memories byte-stable.

The two-seed audit in
`experiments/external_learned_stream_binding_factual_lifecycle/` uses five
anonymous streams, two live identities, three delayed provisional identities,
zero factual replay, zero controller updates, and zero factual optimizer
updates. Seeds `2501` and `2502` both learned the correct joint proposal,
rejected scalar and wrong-held-out replacements atomically, retained the
sibling factual slot, routed the new slot, reloaded exactly, and kept the
binding encoder and controller frozen. A drift control keeps identity matched,
returns an actual factual `conflict`, and leaves the factual-bank content digest
unchanged. Each run records 483 scalar verifier bits: 480 policy-training
outcomes plus three transaction outcomes.

This promotes a bounded joint binding/factual transaction. It does not promote
learned verifier design, autonomous eviction economics, unrestricted memory
growth, arbitrary drift recovery, or general continual learning. The next
pressure point is to vary the factual model family, delay law, and open-set
arrival process while measuring retention and transfer against matched-fresh
controls.

## Joint learned binding and factual-memory growth (2026-08-10)

The external boundary now supports verified append-only growth rather than only
replacement. `ExternalLearnedMultiStreamTransitionContextRouter` grows the
anonymous live-track capacity and the shared factual-bank capacity in one
copy-on-write transaction. A provisional identity is promoted only after its
held-out factual candidate passes, and the complete prior binding/model floor is
byte-stable on the candidate copy. The controller, event encoder, and factual
models remain frozen during deployment; the new knowledge is external state.

The pressure test in `experiments/external_learned_binding_factual_growth/`
uses six-row delayed open-set evidence, two sequential growth transactions,
fresh and outcome-shuffled proposal controls, scalar and wrong-held-out
atomic-rejection controls, exact persistence, and zero replay/controller/
factual-optimizer updates. Across seeds `2601`, `2602`, and `2603`, all gates
passed. The learned policy reached accuracies `1.0`, `0.75`, and `1.0`; fresh
controls reached `0.0`, `0.0`, and `0.0`; shuffled controls reached
`0.375`, `0.125`, and `0.2917`. The first new slot selected the affine
sufficient-statistics family; the nonlinear second stream selected the
random-feature family, demonstrating that the held-out verifier—not the
stream identity—selects the external computation family.

This promotes bounded replay-free external growth with retained prior slots and
learned anonymous proposal ranking. It does not establish unrestricted memory
growth, learned verifier design, arbitrary new computation beyond the registered
candidate families, or general continual learning. The next requirement is to
make growth repeat over more than two additions while measuring stable
retention, transfer against a matched fresh learner, and memory compression.

## Binding-aware factual consolidation (2026-08-10)

The external memory boundary now has a compaction transaction in addition to
growth. `ExternalLearnedMultiStreamTransitionContextRouter.consolidate_factual_slots_verified`
can share physical parameters between two held-out-equivalent factual slots
without merging their anonymous binding tracks or stable slot addresses. The
transaction runs on a full-state copy; a failed equivalence check, unlike model
family, or mutating retention probe leaves binding and factual state unchanged.
Future adaptation remains safe because the shared model detaches copy-on-write
when one slot receives new evidence.

The three-seed audit in
`experiments/external_learned_binding_factual_consolidation/` retained three
opaque streams and reduced physical factual models from three to two on every
seed. It rejected unlike-family consolidation and a probe that attempted to
mutate candidate state, retained all slot addresses and bindings, reloaded
exactly, and used zero replay, zero factual optimizer updates, and zero
controller updates.

This promotes bounded binding-aware factual parameter sharing, not learned
consolidation policy, semantic stream merging, unrestricted memory growth, or
general continual learning. The next pressure should combine repeated growth,
learned maintenance choice, and compression under a finite capacity budget.

## Learned finite-budget maintenance choice (2026-08-10)

The external-memory boundary now has a replaceable discrete maintenance policy
with four actions: `grow`, `share`, `compress`, and `defer`. It consumes only
generic storage telemetry—capacity pressure, slot/alias fractions, lifetime
usage and age, prediction error, binding pressure, redundancy, and compression
opportunity. Structural action masks are supplied by the memory implementation;
the policy cannot make an illegal operation legal, and it cannot commit a
mutation. Existing copy-on-write retention probes remain authoritative.

`ExternalLearnedMultiStreamTransitionContextRouter.propose_maintenance` and
`apply_maintenance_proposal` connect the policy to factual growth, binding-aware
sharing, and runtime compression. Compression now has an explicit
`compress_and_commit_verified` path: the candidate is restored independently,
retention-probed, checked for probe mutation, and only then swapped into the
live bank. `defer` is an explicit no-op rather than an implicit caller branch.

The pressure test in `experiments/external_memory_maintenance_policy/` learns
the action choice from one scalar verifier utility per step, with zero replay
and a frozen controller. Three seeds beat matched fresh and reward-shuffled
controls; the first seed scored `0.50` versus `0.25` fresh and the two audit
seeds scored `0.75` versus `0.25` fresh. All four actions were observed and
the policy state round-tripped by checksum.

This promotes learned maintenance selection, not learned verifier design,
autonomous equivalence discovery, universal continual learning, or unrestricted
memory growth. The next pressure is to train the policy against real verified
retention and byte-cost outcomes while repeating growth, sharing, and
compression over a longer nonstationary stream.

## Real-transaction maintenance audit (2026-08-10)

The maintenance policy has now been tested against actual external-bank
receipts rather than synthetic action labels. The real pressure test performs
retention-verified growth, held-out-equivalent parameter sharing, and
float16 candidate compression; the scalar utility includes the observed
transaction result and byte savings. A matched no-op/defer regime prevents the
policy from treating every available operation as beneficial.

The three-seed archive is
`session_records/sequence_working_memory_2026-08-02/external_memory_real_maintenance_promoted_2026-08-10/`.
All seeds reached `0.95` held-out utility versus `0.70`, `0.7375`, and `0.70`
fresh controls. The action-shuffled controls reached `0.25`, `0.70`, and
`0.2375`, providing a causal check that the learned maintenance choices—not
only exposure to utility—drive the gain. Persistence was exact, compression
saved `5664`, `5472`, and `5616` bytes, and mutating retention probes were
rejected atomically.

This promotes learned maintenance selection over real memory transactions,
not learned verifier design, autonomous equivalence discovery, unrestricted
growth, or general continual learning. The next bottleneck is longer
nonstationary operation where the memory must decide when to retain, replace,
share, or compress under accumulating interference rather than independent
micro-scenarios.

## Long nonstationary maintenance stream (2026-08-10)

The next rung keeps one bank alive for `640` unique verifier utilities instead
of resetting it between scenarios. The versioned maintenance boundary now also
contains `evict`, which uses stable logical slot IDs and a retention-gated
copy-on-write transaction. Across seeds `6120`, `6121`, and `6122`, the stream
repeats growth, equivalent-slot sharing, compression, and safe disposable-slot
eviction while retaining four recurring opaque capabilities.

The promoted archive is
`session_records/sequence_working_memory_2026-08-02/external_memory_long_nonstationary_promoted_2026-08-10/`.
Trained online utility is `0.9953`, `0.9969`, and `0.9969`, versus
`0.5156`, `0.5328`, and `0.5188` for shuffled-verifier controls. Retention
stays above `0.9991`, persistence is exact, replay is zero, and the controller
is frozen. Final held-out utility is reported separately because repeated
future opportunities can eventually repair a shuffled or action-shuffled
policy; the result is a sample-efficiency gain, not unrestricted continual
learning.

The remaining pressure is to remove the predeclared candidate schedule and
make candidate discovery, identity formation, and retention demand arise from
partially observed multimodal experience while preserving the same economics.

## Canonical policy-free execution seam (2026-08-10)

The exported games session exposed a gap in the earlier implementation: a
planner existed, but the production amodal runtime still sent the
controller's direct intention to decoders. That left a stale preferential
policy in the live path even when factual model search was available.

`PolicyFreeAmodalRuntime` closes that gap. The controller updates working
state and exposes one opaque learned state representation; an external goal
state (the destination held by long-term memory) and a runtime-sized set of
opaque candidate intentions go to `ExternalModelBasedPlanner`. The first
planned intention, not `controller.intention`, is sent to the intention bus.
When the planner owns an `ExternalTransitionModelBank`, it performs
goal-conditioned factual retrieval before any caller-owned adaptation.

```text
N encoders -> event bus -> one controller/working memory -> opaque state
                                                        + opaque goal
                                  -> factual model bank -> search
                                  -> intention bus -> M decoders
```

This is a canonical execution boundary, not yet a general capability claim.
The goal representation, candidate intention basis, factual model family, and
verification probes remain replaceable external components. The next
promotion must measure model-free controller behavior against this path on a
nontrivial held-out stream, with zero-shot capability, search expansions,
latency, target updates, lifetime cost, and retention reported separately.

## External opaque intention repertoire (2026-08-10)

Candidate formation is now an independent memory boundary rather than a
caller-owned action list. `ExternalIntentionRepertoire` stores observed
controller-output vectors, merges only near-duplicate opaque entries, and
accumulates verifier/propensity statistics without replay or controller
updates. `PolicyFreeAmodalRuntime` can therefore retrieve a variable-sized
candidate set from external experience and pass it directly to factual model
search.

The boundary includes an important safety rule: an unverified controller seed
is not mixed into verified candidates by default. It remains available as an
explicit exploration option and as the fallback for an empty repertoire. This
prevents unknown output vectors from poisoning beam search while preserving a
path for later verifier-gated acquisition. The promoted audit is archived in
`session_records/sequence_working_memory_2026-08-02/policy_free_intention_repertoire_promoted_2026-08-10/`.

The result is still bounded candidate retrieval, not general continual
learning. The unresolved frontier is learning new intention content and its
representation from partial multimodal experience, then admitting it only
after held-out factual and retention checks.

## Verifier-gated intention admission (2026-08-10)

New output content now has an explicit transactional path. A novel opaque
vector is staged on a copy of `ExternalIntentionRepertoire`; a caller-owned
held-out verifier can validate factual consequences and write the scalar
outcome to the staged entry. The transaction commits only if all retained
entries are unchanged and exactly one candidate was added. Rejected candidates
and verifiers that mutate old entries are complete no-ops.

The promoted audit under
`session_records/sequence_working_memory_2026-08-02/policy_free_intention_admission_promoted_2026-08-10/`
demonstrates one real capability boundary: a goal unreachable with the
existing repertoire is mastered after one verified new intention is admitted,
while a mismatched intention is refused. The controller, factual model, and
old repertoire entries remain stable with zero replay.

This closes safe admission, not candidate invention. General continual
learning still requires an external proposer that can generate useful new
intention content from partial multimodal evidence and an outcome-only active
exploration loop that can discover it efficiently.

## External compositional intention exploration (2026-08-10)

`ExternalIntentionCompositionExplorer` provides a bounded, auditable source of
candidate content from retained opaque experience. It composes pairs of
verified vectors using a versioned operation set (`mean`, `sum`, and
`difference`), removes near-duplicates, preserves source-pair provenance, and
does not mutate the live repertoire. Candidates remain ephemeral until the
held-out `admit_verified` transaction accepts one.

The promoted three-seed audit now derives the diagonal intention from retained
entries `(0, 1)`, then verifies and admits it without controller updates,
replay, or old-entry drift. This is a real improvement in acquisition
provenance, but the boundary remains bounded composition rather than learned
operation discovery, arbitrary new computation, or general continual
learning. The next experiment should drive candidate proposals from partial
multimodal observations and active outcome-only exploration.

## Signed external-entry value factorization (2026-08-10)

The exported games-session frontier exposed a more general delta requirement:
an external entry must be able to reverse an existing value prediction without
forcing the shared state representation to relearn the whole value surface.
`ExternalSignedEntryValueModel` makes that contract explicit. Its state path
produces a positive, polarity-free salience; its opaque entry path produces an
odd scalar polarity; the factual value is their product. Negating an entry
therefore negates the prediction by construction, while a zero entry is
neutral.

The promoted three-seed audit in
`session_records/sequence_working_memory_2026-08-02/signed_entry_value_promoted_2026-08-10/`
trains only on positive entries, freezes the model, and evaluates negative
entries with zero target updates. Entry shuffling breaks the prediction,
exact oddness and persistence hold, and the signed model beats a matched
unfactorized control on the contradictory target. This is a real reusable
signed-delta boundary. `ExternalModelBasedPlanner` and
`PolicyFreeAmodalRuntime` now accept runtime-sized opaque candidate-entry
tensors and an explicit entry-value weight, so the external value model can
change the searched intention without adding a controller or protocol branch.
The planner still leaves arbitrary value learning, changing-regime transfer,
and general continual learning unqualified; the next pressure test is this
live search path across contradictory external regimes without replay.

## Live signed-entry search (2026-08-10)

The planner/runtime seam is now executable rather than merely representational.
`ExternalModelBasedPlanner` accepts a versioned external entry-value model,
runtime-sized `candidate_entries`, and an explicit nonnegative value weight.
At each terminal expansion it evaluates the predicted state with the matching
opaque entry and subtracts that factual value from the search score. The
transition model, controller, and decoders remain unchanged; bank-backed
model selection passes the same boundary through each factual slot.

The promoted three-seed audit under
`session_records/sequence_working_memory_2026-08-02/signed_entry_search_promoted_2026-08-10/`
trains only on positive entries, freezes the external value model, and flips
the selected intention when the external entry assignment is reversed. A
matched planner with no entry-value model is polarity-insensitive. This
promotes live signed-delta search, not arbitrary value learning or general
continual learning. The next pressure is persistent entry growth and
changing-regime search with independent held-out factual verification.

## Persistent external entry repertoire (2026-08-10)

`ExternalEntryRepertoire` is now the independent long-term store for factual
value entries. It grows append-only, deduplicates near-equivalent opaque
vectors, accumulates outcome/propensity statistics without replay, round-trips
through a checksummed payload, and admits novel entries only through an
isolated held-out verifier. `PolicyFreeAmodalRuntime` can retrieve a
runtime-sized proposal and record post-search outcomes without updating the
controller or entry-value model.

This is a real memory lifecycle, not unrestricted memory growth or learned
compression. `ExternalEntryBindingRepertoire` now stores intention↔entry pairs
atomically and `PolicyFreeAmodalRuntime` can propose both tensors from one
record, eliminating positional joins between independent repertoires. The
next pressure is retention-safe consolidation and compression across changing
regimes, with stable logical IDs preserved through maintenance and held-out
factual retention as the admission gate.

## Retention-safe external binding consolidation (2026-08-10)

`ExternalEntryBindingRepertoire.consolidate_verified` adds a copy-on-write
maintenance transaction to the external memory boundary. It combines selected
opaque intention↔entry records into one replacement pair while aggregating
their outcome and exact-propensity sufficient statistics; no old examples are
replayed and the controller remains untouched. A caller-owned held-out
retention probe is authoritative, and a probe that rejects or mutates its
candidate leaves the live state byte-stable.

The transaction keeps stable logical IDs usable after physical compaction:
one retired ID becomes the replacement address and the others resolve through
checksummed aliases. Alias state, statistics, and versioned persistence are
validated on reload. `PolicyFreeAmodalRuntime` exposes the same external
operation without introducing a controller or protocol-specific branch.

The current tests establish the memory invariant only. They do not establish
learned equivalence discovery, autonomous maintenance economics, unrestricted
growth, or general continual learning. The next pressure is a long
nonstationary stream where candidate selection and retention decisions are
learned from verifier outcomes and evaluated with stable-prefix retention and
matched-fresh transfer accounting.

## Stable-address intention memory (2026-08-10)

The same files-like lifecycle now applies to the standalone
`ExternalIntentionRepertoire`, which is the policy-free runtime's fallback
candidate store when no atomic intention↔entry binding bank is configured.
Observed output vectors receive stable logical IDs; physical compaction keeps
one replacement address and persists aliases for retired IDs. Outcome and
exact-propensity sufficient statistics are aggregated directly, so maintenance
does not replay old examples or update the controller.

`consolidate_verified` is exposed through `PolicyFreeAmodalRuntime` and uses
the same isolated retention-probe and mutation-integrity gate as binding
memory. Proposal and composition provenance now report logical IDs rather than
physical positions, so reordering or compaction cannot silently redirect a
durable output reference. Legacy payloads without address metadata remain
loadable and are assigned their original positional IDs.

This generalizes the memory safety contract; it does not create arbitrary new
intention content or establish learned equivalence discovery. The next
capability bottleneck remains outcome-only candidate generation from partial
multimodal experience, with retention and transfer measured against a fresh
learner.

## Outcome-trained continuous intention generation (2026-08-10)

The next part of that boundary is now implemented by
`ExternalOutcomeIntentionGenerator`. It is a replaceable memory-side
stochastic neural program: a learned opaque controller context enters a small
external tanh/linear generator, which samples provisional intention content.
The generator stores its weights, baseline, eligibility traces, and counters
as external state. It has no controller parameters and receives no raw
modality data, task labels, correct actions, or differentiable verifier
signal.

```text
opaque controller context
          -> external Gaussian intention generator
          -> provisional opaque intention
          -> held-out factual verifier
          -> admission into stable intention memory
          -> policy-free factual search
```

Delayed scalar feedback updates the generator with a Gaussian score-function
credit rule. Exact proposal log densities are retained for accounting, while
missing evidence is a no-op. Cells can be appended copy-on-write from a
protected predecessor; protected cells never change their learned content
when later feedback arrives. State persistence is tensor-only and versioned.

The focused causal rung demonstrates that a hidden continuous verifier can
move generated content toward a target from scalar outcomes, while
outcome-shuffled feedback fails, and that missing feedback, protected-cell
retention, copy-on-write growth, and exact reload all hold. This is an
implementation milestone, not a promotion of general continual learning.
The generator is a bounded proposal mechanism, not an arbitrary program
inductor or deployed policy. A generated intention remains provisional until
`ExternalIntentionRepertoire.admit_verified` passes an independent held-out
factual and retention probe. The next experiment must test partial
multimodal contexts, competing old candidates, delayed/noisy outcomes, and
matched fresh-learner transfer with unique verifier bits, optimizer updates,
replay, latency, and stable-prefix retention reported separately.

## Canonical runtime seam and first promoted rung (2026-08-10)

`PolicyFreeAmodalRuntime` now accepts an optional
`ExternalOutcomeIntentionGenerator` and caller-owned generator state. During a
step, the controller still emits only the versioned opaque model state. The
generator may propose one provisional candidate from that state; the runtime
can plan with it immediately, or append it after verified repertoire candidates
when both are configured. The runtime does not mutate generator state. The
caller records the proposal and applies the later scalar verifier outcome
through explicit runtime methods, preserving the separation between inference,
external learning, and durable admission. Atomic intention↔entry bindings
reject generator injection because a standalone candidate has no entry to bind.

The replicated bounded audit is archived in
`session_records/policy_free_intention_generation_2026-08-10/`. With the
controller and state adapter frozen and zero replayed examples, protected
successor cells reached mastery in `9/13` and `11/20` updates relative to fresh
learners; shuffled outcomes failed. This promotes the runtime seam and
external-memory training contract, not general continual learning. The next
gate must challenge the seam with partial multimodal context, delayed or noisy
outcomes, multiple competing memories, reversals, and repeated growth.

## Independent-cell memory and negative-transfer rollback (2026-08-10)

The first generator integration used one external state row per controller
batch row. That was sufficient for a one-cell causal proof but did not provide
memory-sized capacity: a batch of one could not query a growing set of files.
`ExternalOutcomeIntentionMemory` separates those dimensions. It proposes one
opaque candidate per external cell for each adapted controller context, and
`ModelBasedPlanningResult.candidate_indices` preserves the exact candidate
provenance needed for outcome credit. Proposal-specific score gradients are
stored in the ephemeral proposal, so delayed outcomes remain aligned even when
another cell changes before feedback arrives.

The promoted audit is archived in
`session_records/policy_free_intention_memory_2026-08-10/`. It masks half the
opaque context, delays outcomes, adds verifier noise, grows three cells, and
compacts redundant verified output records. A copied cell fails a reversal
probe in both seeds; the candidate transaction is discarded, a fresh cell is
grown, and reversal mastery returns while protected source and successor cells
remain unchanged. This is the first explicit negative-transfer safeguard in
the external learning boundary.

The result promotes independent external capacity and transactional rollback,
not learned routing or general continual learning. The next required step is
to learn which opaque cell to attempt from context and history, rather than
having the lifecycle caller supply the writable cell, then test open-ended
growth and Brain Workshop transfer.

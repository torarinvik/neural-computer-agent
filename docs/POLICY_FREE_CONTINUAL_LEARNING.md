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

## Learned opaque cell routing (2026-08-10)

`ExternalOutcomeIntentionRouter` closes the caller-selected-cell gap. It wraps
the independent-cell memory with a learned context-to-cell softmax, bounded
unseen-cell exploration, delayed route-score credit, and per-cell decision and
feedback counters. The runtime receives only the routed opaque intention; it
does not receive or choose a cell index. The proposal retains the selected
cell, route propensity, and route score gradients so a later scalar verifier
outcome can credit both the emitted cell content and the route that selected
it. Routing now happens before content generation: sparse proposals carry the
physical IDs of only the selected cells, so a one-context step does not
materialize score gradients for the whole memory bank.

The two-seed audit in
`session_records/policy_free_intention_routing_2026-08-10/` uses one frozen
controller, partial opaque context, delayed feedback, source/successor/reversal
copy-on-write growth, protected cells, and a rollback of an inherited reversal
that fails its probe. Both seeds automatically attempt the appended successor
cell without a caller address, recover a fresh reversal under 20% verifier
noise, pass reward-shuffled, action-shuffled, missing-evidence, corruption,
persistence, frozen-core, and zero-replay gates, and retain protected cell
content. The sparse-materialization gate is also one on both seeds.

This promotes caller-free bounded routing over external memory. It does not
promote unrestricted growth, learned compression, arbitrary new computation,
or general continual learning. With a matched fresh cell cloned before source
training, warm successor acquisition is `6.57x` and `2.56x` cheaper in update
count across the two seeds. This is a bounded positive transfer result, not a
claim of unrestricted growth or Brain Workshop mastery. The next high-ROI
pressure is a longer stable-prefix retention/transfer ledger that measures
route and verifier cost as the library grows.

## Route identity is an information boundary

The exported games session identified a useful route-query pattern: a final
state alone can collapse distinct regimes, while trajectory statistics retain
more of the learned evidence needed for addressing. The production runtime now
supports `ExternalControllerTrajectoryQueryAdapter`, which forms an opaque
route query from the controller's final learned representation and masked
mean/max learned event-token statistics. The planner still sees the normal
state representation; only the replaceable external address resolver sees the
richer query.

New external cells also receive an unqualified-cell exploration floor. This
prevents a new file from becoming permanently unreachable before it has
received enough verifier evidence to qualify or be rejected. The floor is not
a semantic route label and does not bypass held-out verification.

`ExternalOutcomeIntentionRouter.verify_and_protect` is the explicit held-out
retention transaction. It accepts only a fresh verifier prefix and commits a
protection bit when the stable prefix clears its floor; it does not update
generator content, route parameters, counters, or eligibility traces. This
keeps online exploration evidence separate from the authority that freezes a
memory file.

The six-regime audit was rejected and is recorded in
`session_records/policy_free_intention_prefix_growth_rejected_2026-08-10/`.
Its failure is architectural evidence: blindly copying an intention-generator
policy into a contradictory regime is negative transfer. The next promotion
must use a factual residual/delta or a fresh challenger, then commit only
after held-out candidate mastery and complete-prefix retention pass.

## Promoted six-regime stable-prefix retention (2026-08-10)

The follow-up uses fresh unprotected challenger cells, eight perturbed
held-out contexts per regime, delayed-feedback settlement before mastery is
declared, and verifier-gated protection. Once a cell passes its held-out
prefix, its content and route state freeze together; its verified context
prototype becomes a memory-side address prior. Relevant reversal evidence is
the only release path.

Both seeds pass all bounded retention gates: six regimes master, every new
cell is held-out qualified, content floors are `0.9596` and `0.9617`, route
floors are `0.5884` and `0.6726`, and all causal, persistence, frozen-core,
and zero-replay controls pass. The reports are archived in
`session_records/policy_free_intention_prefix_growth_promoted_2026-08-10/`.

This promotes bounded stable-prefix retention and address preservation, not
positive transfer. The matched fresh learner remains faster on some disjoint
successors, so the next bottleneck is a verified factual/residual challenger
that improves acquisition without copying policy-like generator weights.

## Promoted one-pass factual residual growth (2026-08-10)

The next pressure test now exercises that challenger. A source transition
model is trained once and frozen. A successor regime is admitted through an
opaque context-addressed random-feature residual whose sufficient statistics
consume each of `32` fresh transition rows once. The candidate must pass an
independent held-out one-step probe, a two-step recursive rollout, and source
retention before copy-on-write promotion.

Seeds `101` and `102` both pass. Residual held-out MSE is `0.000890`/`0.005689`,
recursive rollout MSE is `0.000551`/`0.028136`, and source-retention MSE is
`0.004217`/`0.000352`. Shuffled transition evidence is rejected, missing
evidence is a no-op, persistence is exact, and the frozen base remains
byte-stable. Evidence is archived in
`session_records/policy_free_factual_residual_growth_promoted_2026-08-10/`.

The matched full-model-copy and fresh controls can fit the successor, but
replay the target rows through `1,500` optimizer updates and fail source
retention at target stability. This promotes one-pass factual residual
acquisition, not unrestricted residual capacity, arbitrary program induction,
policy learning, or general continual learning. The next bottleneck is
multi-regime residual growth with route-cost, compression, and bounded-memory
accounting.

## Promoted multi-regime factual residual stream (2026-08-10)

The longer pressure test now admits six distinct factual regimes plus a
reversal into seven opaque residual slots. The shared transition model is
trained once and frozen. Each slot consumes `32` unique transition rows using
one-pass random-feature sufficient statistics, and admission requires held-out
one-step accuracy, a two-step recursive rollout, and complete-prefix
retention of every earlier slot.

Seeds `101` and `102` both promote all seven lifetimes. Maximum retained-prefix
MSE is `0.004544` and `0.009934`; maximum rollout MSE is `0.018566` and
`0.016057`; and source-retention MSE is `0.004217` and `0.000352`. Opaque
route round-trips return slots `0..6` after `21` existing-slot comparisons for
the seven novel bundles. The shared base remains byte-stable, shuffled
reversal evidence is rejected, missing and corrupted evidence are non-mutating,
and persistence is exact.

Float16 compression passes the same held-out residual probe and reduces bank
storage from `125,552` to `62,804` bytes. Int4 is rejected because its
decompressed residual behavior does not retain the prefix. The residual path
uses zero replay; matched fresh controls use `2,400` optimizer updates and
replay `76,800` examples. Evidence is archived in
`session_records/policy_free_factual_residual_stream_promoted_2026-08-10/`.

This promotes bounded factual-memory scaling with verifier-gated growth and
compression. It is still not general continual learning: the bank is bounded,
the context encoder is fixed, and no new arbitrary computation has been
demonstrated. The next bottleneck is scaling beyond a fixed residual basis and
capacity while preserving route identity, compute cost, and retention under
unseen task families.

## Promoted capacity-scaled factual memory and learned reliability (2026-08-10)

The next pressure test admits nine factual regimes plus a reversal into ten
opaque residual slots. The shared transition model remains frozen. After four
slots, a verifier-gated copy-on-write transaction expands capacity from `4` to
`8`; later admissions reach capacity `10` without changing retained content.
Every lifetime consumes `32` unique transition rows and passes held-out
one-step, recursive-rollout, and complete-prefix retention probes.

Seeds `101` and `102` both promote. Maximum prefix MSE is `0.004544` and
`0.014426`; route round-trips recover slots `0..9` after `45` existing-slot
comparisons. A rejected growth proposal is a no-op, shuffled reversal is not
promoted, the frozen base remains byte-stable, and persistence is exact.

An external replay-free error-bin reliability component learns from `142`
verifier outcomes without retaining rows. It allows clean reads with
probability `0.917`/`0.958` and scores corrupted/OOD evidence at `0.250` for
both seeds. Corrupted evidence and a state outside the training range are
rejected without mutating either memory or reliability state. Float16
compression reduces residual storage from `179,360` to `89,720` bytes; int4 is
rejected. Evidence is archived in
`session_records/policy_free_factual_residual_capacity_promoted_2026-08-10/`.

This promotes bounded capacity-scaled factual memory with learned external
reliability. It is still not general continual learning: the residual basis,
context encoder, and verifier calibration remain bounded, and no arbitrary new
computation has been demonstrated. The next bottleneck is scaling and
maintaining these mechanisms under genuinely novel distributions and learned
procedures rather than synthetic factual regimes.

## Factorized masked content and residual reuse (2026-08-10)

The next step makes reusable computation across evidence distributions
explicit. In opt-in masked mode, `ExternalOutcomeIntentionGenerator` now has
a mask-stable content ABI: the observation mask remains in the proposal for
routing and retention, but its channel is structurally disconnected from the
mutable nonlinear content path. A separate learned value-only context
residual can explain evidence-specific output changes without rewriting the
shared hidden content program. Both paths live in replaceable external memory
and receive the same delayed scalar credit; the controller stays frozen.

The two-seed audit in
`session_records/policy_free_intention_masked_routing_factorized_promoted_2026-08-10/`
passes the existing overlapping-mask promotion gates. Warm successor
acquisition takes `9/26` and `11/20` updates against matched fresh learners,
with source/successor held-out retention, shuffled reward/action controls,
missing-evidence no-op, corruption, exact reload, frozen-core, noisy
reversal, and zero-replay controls all passing. Generator schema v2 migrates
older compatible dense files by initializing the residual tensors to zero, so
the factorized path can be enabled as an external-state upgrade.

This promotes factorized external reuse for the bounded overlapping-mask
regime. It does not promote multi-stage evidence growth: the strict
versioned curriculum still reaches its final stage at a fixed update boundary
and has no replicated warm-over-fresh speedup. Unrestricted memory growth,
learned compression, arbitrary new computation, and general continual
learning remain unqualified.

## Adaptive sequential evidence-version growth (2026-08-10)

The fixed-boundary versioned curriculum was replaced by an opt-in adaptive
stage protocol. Each of seven observation-mask versions must reach mastery,
pass an independent eight-outcome held-out prefix verifier, and only then
fork the next protected external cell. The fork copies both the reusable
factorized content and the route key on previously observed dimensions; the
sole unqualified child receives a `0.75` exploration floor so caller-free
routing can discover it without a task-specific cell index. Each stage has a
four-update minimum, and the fresh control uses the identical protocol.

The three-seed audit is archived in
`session_records/policy_free_intention_masked_routing_adaptive_promoted_2026-08-10/`.
All seeds pass stage completion, source/successor retention, caller-free
routing, reversal, shuffled reward/action, missing-evidence, corruption,
persistence, frozen-core, and zero-replay gates. Warm/fresh successor
acquisition is `39/50`, `42/44`, and `34/55` updates, for transfer ratios
`1.282`, `1.048`, and `1.618`; each run grows nine external cells and spends
`128` held-out stage-verifier bits.

This promotes bounded adaptive sequential reuse across a known seven-stage
evidence curriculum. It does not establish arbitrary distribution shift,
unrestricted growth, learned compression, arbitrary new computation, or
general continual learning. The next pressure is novel mask ordering and
unseen evidence combinations rather than another fixed schedule.

## Verifier-selected copy-or-fresh admission (2026-08-10)

The next boundary is negative-transfer safety on an unseen task. The external
router now exposes `select_verified_transfer_prior`, an isolated transaction
that creates a copy-on-write candidate inheriting a protected source cell and
a fresh candidate with the same frozen controller and state adapter. A
bounded outcome-only probe scores both branches; only the winning branch and
its post-probe state are returned. The live source state is digest-checked and
cannot be mutated by the challenger.

The direct-copy novel-task baseline is negative transfer on all three seeds:
warm adaptation takes `63/68/85` updates versus matched fresh `34/26/61`.
The promotion-quality challenger rejects copied state in all six warm/fresh
decisions, then all selected fresh branches master an unseen evidence mask and
target at scores from `0.9802` to `0.9839`, with held-out retention and post-reversal retention. The
audit also covers shuffled-outcome, missing-evidence, memory-corruption,
exact-reload, frozen-core, and zero-replay controls. Reports are archived in
`session_records/policy_free_intention_novel_challenger_promoted_2026-08-10/`.

This promotes bounded verifier-selected copy-or-fresh external intention
admission. It does not claim positive transfer on novel tasks, arbitrary new
computation, unrestricted memory growth, compression, or general continual
learning. The next high-ROI experiment is a probe that sometimes accepts a
useful transfer prior on a genuinely novel combination, followed by learned
cost-aware admission and retention across multiple unseen task families.

## Positive prior selection on a nearby unseen task (2026-08-10)

The complementary audit uses the same verifier transaction with an unseen
mask and an unseen target that is a nearby continuation of the mastered
successor. Transfer wins all six warm/fresh challenger decisions with a probe
margin of at least `0.094`, and every selected transfer branch masters and
retains the novel capability. The evidence is archived in
`session_records/policy_free_intention_positive_transfer_challenger_promoted_2026-08-10/`.

This is the first replicated positive prior-selection signal: the verifier
can accept useful inherited external computation as well as reject harmful
copying. Matched warm acquisition is faster in two of three seeds, so this
does not establish a universal warm-over-fresh speedup or broad positive
transfer. The next step is cost-aware admission across multiple genuinely
different task families.

## Sequential cost-aware admission and prefix retention (2026-08-10)

The next audit admits three unseen families sequentially from one protected
successor: nearby positive transfer, unrelated negative transfer, and an
alternate nearby successor. Each admission uses nonzero cost-aware v2
selection, then a held-out verifier checks the complete prefix of all earlier
files before the next file is created. Across three seeds, every run selects
`transfer -> fresh -> transfer`, grows external memory from eight to eleven
cells, and passes every prefix verifier. Reports and accounting are archived
in `session_records/policy_free_intention_sequential_admission_promoted_2026-08-10/`.

This promotes bounded repeated admission with complete-prefix retention and
explicit deployment-budget accounting. It does not establish broad task-
family generalization, arbitrary new computation, unrestricted growth,
compression, or general continual learning. The next pressure is a learned
cost model and a larger task-family matrix without relying on a fixed source
cell or synthetic target geometry.

## Automatic verified source retrieval (2026-08-10)

The sequential audit no longer passes a physical source-cell index into the
copy challenger. `ExternalOutcomeIntentionRouter.select_verified_source_cell`
chooses only among protected, held-out-verified cells using learned route
compatibility, verified prototypes, observed support, quarantine state, and
mask profiles. The selected source and candidate scores are returned in a
versioned receipt, while the controller remains unaware of the address.

Across three seeds, automatic source retrieval preserves the sequential
`transfer -> fresh -> transfer` result, complete-prefix retention, and all
causal controls while selecting a different verified stage cell in one warm
run. Evidence is archived in
`session_records/policy_free_intention_sequential_auto_source_promoted_2026-08-10/`.

This removes a real caller-side lifecycle leak, but the selector is still
bounded by the current learned context/prototype space. The next pressure is
to learn source costs and compatibility over a larger, non-synthetic task
family stream.

## Learned source/admission cost (2026-08-10)

The next increment moves the remaining hand-specified transfer/fresh costs
behind a replaceable memory-side contract:
`ExternalRoutedIntentionCostModel`. It receives only masked opaque context,
verified source coverage, and the current external-cell count. A completed
admission contributes one normalized continuation-cost observation to the
branch that was actually selected. The model retains sufficient statistics,
not task IDs, trajectories, or replayed examples; the controller, state
adapter, and factual content remain frozen.

The three-seed audit is archived under
`session_records/policy_free_intention_learned_cost_promoted_2026-08-10/`.
Every run passes automatic source selection, cost-aware v2 receipts, mastery,
complete-prefix retention, reversal, corruption, missing-evidence, causal
controls, exact model-state reload, frozen-core, and zero-replay gates. Warm
runs select `transfer -> fresh -> transfer`; the matched-fresh run for seed
85302 selects `fresh -> fresh -> transfer`, showing that the learned policy
does not force a historical sequence when the outcome-only challenger favors
fresh initialization.

This promotes a bounded learned admission-cost contract and removes a
caller-side economic schedule. It does not yet show broad cost prediction or
acquisition gains: the current probe scores dominate the decision, and the
stream remains a small synthetic family matrix. The next pressure is a larger
and genuinely non-synthetic family stream in which predicted cost can be
tested against held-out acquisition curves.

## Canonical external computation runtime seam (2026-08-10)

The CPU-plus-files architecture now has a first-class execution boundary.
`ExternalProgramAmodalRuntime` runs one frozen `AmodalCognitiveController`,
one replaceable `ExternalCapabilityRegisterMachine`, and the ordinary
intention bus as one `INPUT -> PROCESS -> OUTPUT` cycle. A portable
`ExternalProgramArtifact` is observed through learned controller state and
opaque action/outcome feedback, executed copy-on-write, and converted into the
intention delivered to decoders. The controller's own intention is retained as
diagnostic state and is not silently decoded.

The external execution snapshot records the observed persistent register, the
transient result, an opaque positional trace, and an artifact checksum. A
failed verifier can therefore discard a candidate file without rolling back or
replaying the controller. `ExternalSequenceProgramMemory` can route among
multiple opaque program files from a replaceable state adapter; the controller
never receives the physical slot address or interpreter metadata.

This closes a real architectural gap exposed by the exported session: the
system can now store executable computation as portable external state rather
than treating artifacts as storage-only or adding a controller branch for each
new capability. It is an execution and replacement contract, not evidence of
learned program synthesis, arbitrary computation acquisition, unrestricted
growth, or general continual learning. The next causal rung is outcome-only
program-file acquisition on held-out Brain Workshop families, with fresh-file,
no-agent, replay, retention, and frozen-controller controls.

## Transactional executable-file admission (2026-08-10)

`ExternalSequenceProgramMemory.admit_verified_artifact()` now provides the
missing copy-on-write file transaction. It validates a portable opaque program
against the shared interpreter ABI, consumes only deterministic scalar
verifier outcomes, and commits the candidate only when a stable verifier
prefix clears the configured threshold. Rejected candidates do not change the
bank. Committed files may be protected independently, while the bank's
artifacts, router tensors, output ABI metadata, protection state, and checksum
round-trip through a tensor-only payload.

This strengthens the CPU/files foundation for frozen-core learning and
prevents candidate failure from becoming silent catastrophic forgetting. It
is still a storage and admission contract: candidates are supplied by an
external learner, so this does not claim program induction, arbitrary new
computation, or general continual learning. The required next experiment is a
held-out Brain Workshop file-acquisition audit with a fresh-file challenger,
no-agent and shuffled-outcome controls, zero controller updates, and complete
prefix retention.

## Promoted executable-file admission and external route cells (2026-08-10)

The executable-memory contract has now passed a three-seed outcome-only audit
(`23001`, `23002`, `23003`). A candidate file is appended transactionally only
after a stable verifier suffix of at least `32` scalar outcomes clears the
threshold; corrupted candidates are rejected as no-ops. The controller and
interpreter are frozen, and no replayed verifier rows are retained.

The target route is held in a separate opaque external cell from the mastered
source route. This is an important correction to the naive “append another
column to one shared policy” design: that design produced measurable route
interference. External cells preserve earlier routes and can grow outside the
controller while the cell selector remains part of replaceable memory-side
state.

The promotion shows bounded verifier-gated executable-file admission and
retention, not program synthesis, arbitrary new computation, unrestricted
memory growth, or general continual learning. The next pressure is to generate
candidate files from scalar outcomes and test the same contract on a larger
non-synthetic Brain Workshop family stream.

## Promoted outcome-only executable-program search (2026-08-10)

`ExternalProgramCandidateSearch` now supplies candidates rather than requiring
the caller to hand over a complete new executable file. It performs generic
opaque sequence edits and learns aggregate operator statistics from scalar
verifier outcomes. Proposals are copy-on-write; only the stable-prefix winner
is passed to `ExternalSequenceProgramMemory.admit_verified_artifact()`.

Across seeds `23001`, `23002`, and `23003`, a protected parent synthesizes a
held-out two-instruction composition in `1--13` proposals. The fresh parent
control fails within `256` proposals, while source and target retention remain
perfect. Corruption, shuffled outcomes, exact reload, frozen-interpreter,
frozen-controller, zero-replay, and zero-controller-update gates pass.

This is bounded one-edit structural synthesis, not open-ended program
induction or general continual learning. The next pressure is a persistent
multi-step hypothesis frontier on a genuine Brain Workshop family stream.

## Persistent multi-step executable hypothesis frontier (2026-08-10)

`ExternalProgramHypothesisFrontier` extends the one-edit search without moving
computation into the controller. Provisional candidate files are copy-on-write
frontier state; the protected root is retained across every update, and
frontier persistence contains only opaque tensors, parent/depth structure,
candidate checksums, and aggregate scalar-outcome statistics. The default
proposal schedule is finite breadth-first opaque expansion, which gives the
multi-step search a deterministic completeness bound for a supplied bank while
remaining compatible with a future learned proposal policy.

Across seeds `23001`, `23002`, and `23003`, a useful parent reaches a held-out
three-step executable target in `22`, `28`, and `13` verifier evaluations. A
matched random parent needs `62`, `66`, and `50`. Source retention and target
mastery are `1.0000` in every run; corruption is rejected without a memory
write; frontier/file reload, protected-root retention, frozen interpreter,
frozen controller, and zero replay all pass. The complete promotion record is
under
`session_records/sequence_working_memory_2026-08-02/external_program_hypothesis_frontier_promoted_2026-08-10/`.

This promotes a bounded multi-step external-memory search and admission seam.
It does not promote arbitrary program induction, unrestricted memory growth,
or general continual learning. The next required pressure test is to use the
frontier against a non-synthetic Brain Workshop family stream and measure
whether newly admitted executable files improve held-out learning curves
without replay or loss of earlier families.

## Verifier-gated executable-memory lifecycle (2026-08-10)

The executable file store now supports a bounded memory lifecycle in addition
to admission. Stable opaque logical IDs survive physical compaction, and
copy-on-write transactions can evict an unprotected file, consolidate a
held-out-equivalent duplicate, or compress durable storage. Each transaction
returns a versioned metadata-only receipt with source/candidate digests and
storage accounting. A rejected transaction must leave the live source digest
unchanged.

The promotion record is
`session_records/sequence_working_memory_2026-08-02/external_program_memory_lifecycle_promoted_2026-08-10/`.
Across three seeds, protected eviction and non-equivalent consolidation are
rejected, equivalent duplicates are compacted, logical identity is retained,
corrupted compressed payloads are rejected, and a deliberately mutating
retention probe cannot commit. Decompressed float16 storage retains the
held-out executable behavior and cuts durable state from `28,032` to `14,016`
bytes. The controller and interpreter remain frozen, with zero replay and
zero controller updates.

This is an important reliability layer for “learn while frozen”: external
memory can grow, age, consolidate, and shrink without silently destroying
older capabilities. It is still not a learned maintenance policy, learned
compression, unrestricted memory growth, arbitrary new computation, or
general continual learning. The next high-ROI test is to make maintenance
choice itself outcome-driven over a longer nonstationary Brain Workshop
stream, charging the verifier and storage costs and comparing against a fresh
memory control.

## Learned maintenance over executable files (2026-08-10)

That next seam is now implemented. `ExternalSequenceProgramMemory` exposes
generic telemetry, a structural action mask, and proposal/application methods
for the replaceable `ExternalMemoryMaintenancePolicy`. The policy learns from
one scalar transaction utility at a time; the file store owns opaque logical
IDs, candidate artifacts, equivalence probes, retention probes, and all
copy-on-write commits.

The promoted three-seed record is
`session_records/sequence_working_memory_2026-08-02/external_program_memory_maintenance_promoted_2026-08-10/`.
The learned policy reaches perfect held-out phase utility on every seed,
beats both fresh and shuffled-verifier controls, observes real grow/share/
compress/evict transactions, and uses zero replay and zero controller updates.

This is a bounded lifecycle-policy result. It does not yet show that the
policy learns when to retain arbitrary newly acquired capabilities under a
long nonstationary Brain Workshop stream, nor that it solves general
continual learning. The next experiment must combine executable hypothesis
frontiers, maintenance cost, and complete-prefix retention on genuinely
rendered families.

## File-scoped execution state (2026-08-10)

The executable runtime now isolates recurrent register/context state by the
external file's stable logical ID. A capability can be revisited after other
files have run without inheriting their working state, while verified file
retirement removes the corresponding state entry. Alternating-route and
retirement/reload tests pass without changing controller parameters.

The same runtime now handles mixed batch schedules: rows belonging to
different external files are executed under row masks in one tick, and the
per-file recurrent state bank is updated only for its assigned rows. This is
important for measuring multiple Brain Workshop families together rather than
silently forcing a single-family batch.

Executable-memory routing now uses the post-step learned event trajectory by
default: the replaceable address adapter receives masked mean/max statistics
over the current event window in addition to the controller representation.
Final-state-only routing remains an explicit compatibility control. This
strengthens route identifiability without exposing raw modality formats or
logical file IDs to the controller.

The runtime now also returns the opaque route query and soft per-file
probabilities outside the controller boundary. A future route learner can use
those values with the observed scalar outcome and exact propensity, including
in a mixed batch, without retaining raw verifier rows or adding a controller
branch. The current runtime does not silently update route parameters; the
credit learner remains an independently versioned memory-side component.

Executable route exploration is now an explicit runtime control rather than
an accidental side effect: greedy deployment reports propensity one, while an
epsilon mixture samples new files and records the selected probability per
row. This supplies evidence to a future route learner without changing the
controller or pretending that exploration itself is learned.

The optional outcome-only route learner is now wired into the executable
runtime. It applies delayed scalar feedback to an external eligibility/policy
trace, then selects the next opaque file from the current learned trajectory;
the controller and interpreter remain frozen. Router state is checkpointed,
and one newly admitted file can be activated through an append-only transaction
without resizing existing computation. The focused causal test shows the
external route policy changes toward the rewarded file while the controller
parameters remain unchanged. This remains bounded route adaptation over
pre-admitted files, not arbitrary program acquisition or general continual
learning.
The runtime also fails closed if file eviction or compaction leaves the
append-only route policy out of sync, preventing silent address reassignment.

The controller-plus-file working state is also restartable. A versioned
tensor-only checkpoint preserves the controller event window and workspace
together with every logical file's recurrent register state, while executable
artifacts and model parameters remain separate resources. Exact mixed-batch
resume and unknown-schema rejection are covered by tests. This removes a
restart-induced source of forgetting, but does not itself establish new-skill
acquisition or general continual learning.

This is a prerequisite for honest continual-learning measurements: content
retention is not enough if temporal working state is shared accidentally. The
next pressure remains a real nonstationary Brain Workshop acquisition stream,
where this isolation must support new capability admission and maintenance
without loss of earlier families.

## Opaque goal-fragment memory and compositional destinations (2026-08-11)

The exported architecture work exposed a missing half of the CPU-plus-files
boundary: factual transition models were external, but destinations were still
passed as caller-owned raw tensors. `ExternalGoalFragmentMemory` now stores
versioned opaque destination fragments independently of the controller. Each
fragment contains only a learned/verified target vector and a boolean mask; no
coordinate is assigned a semantic name and no task identifier is persisted.

`ExternalGoalFragmentSet` composes runtime-sized fragments by either `union`
(satisfy any fragment) or `intersection` (satisfy every fragment). The factual
planner scores masked fragment distance and derives the intention sequence at
inference time. This directly implements the useful “replaced puzzle piece”
formulation: a new destination can constrain only the small part that differs,
while the transition model and controller remain shared. The memory supports
copy-on-write held-out admission, checksum persistence, and true rejected
write no-ops.

`PolicyFreeAmodalRuntime` can now read fragments by opaque memory indices and
pass the composed destination to either a single factual model or a factual
model bank. The planner still has no stored task policy, and the controller
still receives no goal address or composition metadata. This is a destination
memory/composition contract, not evidence of learned goal discovery,
unrestricted memory growth, or general continual learning. The next pressure
is to learn or verify fragment admission from rendered experience and measure
whether intersection fragments reduce acquisition cost on genuinely novel
task families.

## Outcome-only goal-fragment staging (2026-08-11)

`ExternalGoalFragmentStager` closes the first part of that acquisition seam.
A caller can propose an opaque destination candidate from a learned terminal
state and feed the stager one fresh deterministic scalar verifier outcome at a
time. The stager retains only the candidate tensor pair and sufficient
statistics: eligible observation count, cumulative outcome, prefix mean, and
the minimum stable prefix mean. It stores no event rows, outcomes, task IDs, or
replayable trajectories.

Admission remains two-phase and copy-on-write. A candidate must clear the
configured stable prefix before `ExternalGoalFragmentMemory` is asked to run
the independent held-out retention probe. A rejected candidate leaves both
the durable memory and the staging state unchanged; an accepted candidate is
removed from pending staging only after the durable fragment commits. The
runtime exposes this as `observe_goal_fragment()` and
`admit_goal_fragment_verified()`, with the controller, event bus, factual
model, and decoders untouched. `goal_fragment_candidate_from_controller_output()`
and `observe_goal_fragment_controller_output()` provide the stricter path: a
candidate is projected through the runtime's replaceable state adapter before
staging, so planner-space and controller-space tensors cannot be confused by
the caller.

This is an implementation boundary and a replay-free acquisition pressure
test, not a claim that Brain Workshop has learned arbitrary goals. The next
experiment must derive candidates from real rendered Brain Workshop
transitions, compare fresh versus inherited fragment proposals, and charge
every verifier bit while auditing missing-evidence, shuffled-outcome,
reversal, and complete-prefix retention controls.

The bounded harness for that next rung is
`experiments/brainworkshop_canonical/goal_fragment_staging.py`. It trains only
the existing external reader/decoder path, derives a candidate from the
rendered cue's learned event, feeds fresh episode scores into the stager, and
checks missing-evidence, fresh-candidate, inverted-outcome, and reversal
no-admission controls. Its report labels itself `staging_boundary_only`; it is
intentionally not a promotion record until the fragment is coupled to
downstream model-based behavior.

## Context-conditioned goal routing (2026-08-11)

The next coupling is now explicit. `PersistentOpaqueContextRouteEvidence` can
learn which append-only goal-fragment slot is useful for an opaque learned
controller state, using only attempted slot IDs and deterministic scalar
outcomes. `PolicyFreeAmodalRuntime` synchronizes route-slot width with goal
memory, selects a per-batch fragment without caller-supplied addresses, and
passes that fragment into the ordinary factual planner. `propose_per_batch()`
keeps simultaneous contexts separately bound instead of broadcasting one
destination across the batch.

This makes the path causal while preserving the CPU/files boundary: the
controller weights are untouched, route evidence is independently versioned
and serializable, and the planner still derives intentions by model search
rather than reading a stored action policy. Protected route preferences can
also be demoted after a configured reversal prefix; the durable fragment is
not deleted merely because its current context route became stale.

The focused controls cover learned-slot selection, append-order fallback for
unseen contexts, reversal demotion, per-batch binding, and frozen-controller
checks. This is stronger than destination storage, but it remains bounded
context-conditioned routing. It does not yet demonstrate that a fragment can
be discovered from arbitrary rendered experience, that a frozen core can
invent new computation, or that Brain Workshop acquisition improves on a
fresh learner. Those claims still require held-out acquisition curves and
complete-prefix retention controls.

## Opaque transition-observation bridge (2026-08-11)

The policy-free runtime now exposes `transition_observation(output,
successor)`. It packages the current planner-space state, the intention that
was exposed to the decoder, and the next planner-space state into the existing
`ExternalTransitionObservation` schema. A caller may attach only a generic
confidence scalar. Raw keypresses, rewards, task names, and verifier answers
are deliberately absent.

This is the correct training seam for external affine, random-feature, or
neural transition-model banks: collect a fresh row once, update only the
selected external slot through `learn_transition_once()`, and preserve the
controller. That method rejects replay-dependent neural banks, making a
zero-replay claim mechanically explicit. It does not itself prove
that the learned model transfers to a new task; that still requires recursive
held-out rollout error, planner success, fresh-learner comparison, and
complete-prefix retention.

## Rendered replay-free transition acquisition (2026-08-11)

`experiments/brainworkshop_canonical/replay_free_transition_acquisition.py`
now exercises the bridge on real rendered `NBackVerifier` lifetimes. The
controller, event frontend, and keypress decoder are frozen. Consecutive
policy-free outputs become opaque transition rows; an affine external bank
consumes each row once, with no optimizer replay. A matched fresh bank provides
the held-out control, and recursive model error is measured after learning.

The default three-lifetime rung produced `0.02737` trained held-out rollout
error versus `0.06614` for the fresh bank, with `18` transition rows consumed
once, `0` replayed examples, `0` optimizer updates, `16` unique verifier bits,
and an unchanged controller. This is the first rendered evidence that the
external factual model can improve from frozen-core experience. It is still a
transition-model boundary result: it does not yet show new-task acquisition,
goal discovery, planner success, or retention across a nonstationary family.

The same harness now includes a two-family nonstationary rung. Source n-back-2
experience is learned under one opaque rendered cue, then target n-back-3
experience is learned under a different cue/context and isolated bank slot.
The source slot remains byte-stable while the target slot improves its held-out
recursive error over a fresh target bank. The default run measured source error
`0.03704 -> 0.03704`, target error `0.01334` versus fresh `0.04522`, `24`
transition rows consumed once, and zero replay. This is the first direct
no-replay retention result across a nonstationary rendered family, but it is
still isolated factual-model growth; goal discovery and end-task acquisition
remain unqualified.

The next rung removes the target context from the acquisition path. With only
the source slot present, the target rendered lifetime initially behaves through
the known source model; its opaque transition rows are routed to the online
external context router. The router admits a new target slot, consumes the
six-row bundle once through affine sufficient statistics, and the following
target lifetime and held-out lifetime both recover that slot. The default seed
measured target error `0.02440` versus fresh `0.08137`, source error
`0.37234 -> 0.37234`, `24` one-pass rows, and zero replay. A second seed also
passes this boundary. A nearby seed fails the target held-out gate, so this is
not promoted as a stable capability gain: the remaining problem is reliable
online discovery and representation conditioning across seeds, followed by
goal-conditioned end-task acquisition.

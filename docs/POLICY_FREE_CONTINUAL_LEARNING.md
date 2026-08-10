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

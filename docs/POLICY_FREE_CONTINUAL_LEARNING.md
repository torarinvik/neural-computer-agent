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

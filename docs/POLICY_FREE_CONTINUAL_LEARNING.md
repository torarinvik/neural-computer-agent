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

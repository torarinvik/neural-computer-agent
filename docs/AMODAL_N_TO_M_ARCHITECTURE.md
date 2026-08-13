# Amodal N-to-M architecture

This document is normative. Experimental files describe measurements; they do
not redefine the target system.

## Objective

Maximize verified reusable capability per unique experience. New experience
should create external computation that lowers the acquisition cost of later,
mechanistically different tasks.

The target boundary is:

```text
N encoders -> amodal event bus -> fixed controller and workspace
           -> intention bus -> M decoders
                            -> external program and memory system
```

## Hard invariants

1. Encoder and decoder counts vary at runtime without resizing the controller.
2. The controller consumes learned event tensors, never raw modality formats.
3. The controller emits learned intentions, never device or protocol formats.
4. Encoders, controller, memory, programs, and decoders have independent,
   versioned interfaces.
5. Simultaneous streams remain separately bindable. They are not blindly
   averaged into one observation.
6. Latent meanings emerge from verified experience. Coordinates and semantic
   fields are not assigned meanings by hand.
7. Adding an adapter cannot add a modality-specific reasoning branch.
8. Durable task content lives in external memory or programs, not in the fixed
   controller.
9. A content-addressed miss is unknown and fails closed; it is not interpreted
   as a zero-valued prediction.
10. The verifier supplies only deterministic scalar outcomes to the learner.

## Three typed layers

The architecture separates operations that are often conflated.

### Perception reducers

Encoders turn rendered sensory streams into learned event collections. Generic
reducers may discover temporal relations, object persistence, value-bearing
targets, or other regularities, but must not expose task IDs, coordinates,
correct actions, or verifier-private state.

### State operators

A state operator transforms typed learned state:

```text
StateOp : State -> State
```

Operators may be learned, searched, or composed and are stored as independently
verified external artifacts. Bind context once, then execute fixed computation.
Apply structured programs one piece at a time rather than predicting the whole
composite in one shot.

### Executive operations

The executive connects persistent programs to interaction:

```text
ExecutiveOp : Machine x EventBus -> Machine x IntentionBus
```

The minimal target kernel is:

- `RECEIVE(query, destination)`;
- `EMIT(source)`;
- `WAIT`;
- `HALT`;
- typed `READ`, `WRITE`, and `COPY`;
- `CALL(operator_handle, arguments)`;
- `SEQUENCE`, `BRANCH`, `LOOP`, and `RETURN`.

Input/output operations belong to the executive boundary, not to the ordinary
state-transition algebra.

## Outcome and reward events

Reward is an authenticated input event, not a reward-specific cognitive
instruction and not a freely queryable oracle.

The causal interaction contract is:

1. the controller emits an intention;
2. a decoder executes it and the runtime records the exact propensity;
3. the runtime creates an opaque action receipt;
4. the environment emits zero or more trusted outcome events linked to that
   receipt or to an explicitly unknown causal window;
5. `RECEIVE` consumes each event according to queue semantics.

An empty queue means "not observed yet." It does not mean zero reward. Outcome
events carry an opaque payload, trusted source key, timestamp/delay, presence,
confidence, and causal receipt key. Programs and decoders cannot write to the
trusted verifier queue.

## Instruction-set policy

Boolean gates are useful library programs and hardware accelerators, not the
machine's cognitive foundation. A bitwise gate ISA is complete but badly
matched to learned events, objects, values, temporal state, and interaction.

The stable abstraction is `CALL` over a verified operator handle. Frequently
reused operators—including Boolean, arithmetic, relational, temporal, and
geometric functions—may be promoted as macros when they causally reduce
held-out acquisition cost. Promotion must not change program semantics.

Conditions return typed evidence rather than an unqualified bit:

```text
Evidence = {present, value_or_score, confidence}
```

`BRANCH` must handle true, false, and unknown explicitly.

## External programs and memory

The fixed controller is the interpreter and executive. The external bank is
the growing source of durable capability. It may contain:

- factual transition models;
- learned state operators;
- multi-step programs and control-flow fragments;
- goal/value bindings;
- episodic records and working-memory history;
- routing and lifecycle state.

Programs are proposed by a learned reader or search, checked against fresh
experience, repaired only where checks fail, and admitted transactionally.
Unverified programs remain quarantined. Protection, eviction, compaction, and
replacement are verifier-gated and copy-on-write.

Parallel transactional state updates are preferred when outputs depend only on
the pre-state. Sequential execution is used when a real data dependency or
persistent phase requires it. Search cost must be reported by family because
an inexpressible target can look exactly like slow search.

## Learner-visible information

Allowed frontend streams:

- rendered vision, audio, or text;
- the agent's own opaque actions and exact propensities;
- its latent state, workspace, and external memory;
- deterministic scalar verifier outcomes.

Forbidden learner-visible information:

- semantic task or rule IDs;
- game state, coordinates, velocities, or object labels;
- correct or unattempted-action labels;
- English reasoning traces;
- handwritten symbolic solvers;
- private probe metadata.

## Promotion standard

A capability is promoted only when fresh held-out evidence shows stable
mastery, causal necessity, retention, and benefit over a matched fresh learner
when transfer is claimed. Required accounting and controls are defined in
`AGENTS.md` and `PROMOTION_FIREWALL.md`.

## Current boundary

The retained Brain Workshop evidence qualifies bounded append-only external
working-memory computation through n-back-32 with frozen source retention and
zero replay. It does not yet qualify autonomous general program induction,
cross-mechanism maintenance-policy transfer, unrestricted memory growth, or a
complete interactive executive ISA.

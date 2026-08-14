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

`INPUT` is a variable-port executive instruction. Sensory frontends and trusted
verifier adapters implement the same polling ABI and may be attached or removed
at runtime without resizing the controller. Polling concatenates simultaneous
learned events as separate source-bindable records and drains attributed
outcomes exactly once. The minimal verifier port accepts only the agent's own
action receipt, a deterministic scalar, presence/confidence, and observation
time. Port handles are transport metadata and never become controller features.

The causal interaction contract is:

1. the controller emits an intention;
2. a decoder executes it and the runtime records the exact propensity;
3. the runtime creates an opaque action receipt;
4. the environment emits zero or more trusted outcome events linked to that
   receipt or to an explicitly unknown causal window;
5. `RECEIVE` consumes each event according to queue semantics.

Resolved outcomes are offered in the same tick to registered external learners,
including a provisional program learner or the route ledger for the program
that was actually selected. Observers receive only the attempted receipt and
its scalar result; they cannot inspect unattempted programs or infer a task ID.

An empty queue means "not observed yet." It does not mean zero reward. Outcome
events carry an opaque payload, trusted source key, timestamp/delay, presence,
confidence, and causal receipt key. Programs and decoders cannot write to the
trusted verifier queue.

## Live cognitive tick

The deployed runtime is an event-driven cognitive game loop, not an
episode-returning trainer. One monotonic tick executes:

```text
RECEIVE -> resolve outcomes -> bounded online update -> think -> EMIT
```

Input sampling rate, learned-event rate, controller tick rate, optimizer update
rate, and action rate are measured separately. A frontend may sample a screen
at high frequency while emitting no event for unchanged frames. Conversely, a
single tick may preserve several simultaneous source-keyed events. An empty
event collection is a valid quiet tick and never fabricates sensory evidence.

Each emitted action is recorded with its exact propensity, timestamp, output
device key, and model version. The runtime retains the private credit state
needed to resolve delayed evidence, but the public receipt contains only
learner-visible action metadata. Outcome delivery is exact-once. Unknown,
duplicate, or temporally impossible receipt keys fail closed.

For a human-parity physical device, a scalar outcome must also carry the digest
of every public frame used to derive it. The physical Brain Workshop adapter
accepts only explicit visible feedback colors over a complete trial window.
Under default Brain Workshop scoring, green is positive, red/blue is negative,
and a neutral true-negative is absent rather than rewarded. Hidden correctness,
session files, source hooks, and synthetic per-trial rewards are prohibited.
Missing public evidence remains absent rather than becoming a fabricated zero.

"Live" constrains causal order rather than wall-clock pacing. The same tick
implementation may run against an accelerated virtual clock for research or a
real monotonic clock for physical screen/audio/keyboard interaction. Each
logical learner still consumes one causal stream and updates only from newly
received evidence.

Simultaneous live sources are retained as distinct event records and distinct
temporal histories. A generic shared reader may condition on opaque source keys
before composing results, but it may not merge raw event tensors or add a
modality-specific reasoning branch. Device enumeration order is not a semantic
binding and must not change behavior.

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

Temporal-address files use the same lifecycle explicitly. Candidate optimizer
state remains provisional and outside executable memory. Stable public
per-lifetime outcomes gate admission of the learned instruction tensor, which
is bound to the compatible frozen-controller digest. An external memory-side
router then selects immutable files from learned event vectors and updates only
from the attempted slot and its scalar outcome. Unknown contexts remain
exploratory; neither admission nor retrieval accepts a semantic task/rule ID.
The bank and each artifact are checksummed and fail closed on corruption.

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

The first typed executive vertical slice is implemented as an independently
versioned external interpreter. It preserves complete learned event
collections in a typed slot, persists workspace and instruction position
across ticks, calls opaque versioned operator handles, distinguishes missing
evidence from false evidence, and emits only standardized intentions. Invalid
types, empty reads, unknown handles, divergent batched control flow, and step
budget exhaustion fail closed. Operator state is now explicit, independently
versioned, per-executive, and transactionally replaced only after both result
and next-state validation. A generic relative-delay operator demonstrates a
four-tick 2-back relation: two missing-history waits followed by correct match
and mismatch intentions; erasing only temporal presence removes the decision,
and a second executive sharing the same frozen operator objects starts with
clean history. `WAIT` and `EMIT` can yield directly to a validated next target,
which closes the persistent game-loop cycle without consuming an input tick.
This establishes the executable and causal mechanics of stateful composition,
not controller-driven program construction, autonomous `.bank` admission, or
physical deployment of the new interpreter. The v1 program counter is
batch-uniform because the live target is batch one.

The first positive executive-composition transfer is now promoted at a bounded
diagnostic scale. A verified one-step temporal-equality program contributed its
generic relation fragment to a held-out 2-back search, leaving only four opaque
relative-delay bindings to probe. A matched empty-bank learner searched the
same bindings crossed with four generic relation fragments. Across two
disjoint 16-seed blocks, both arms admitted the identical executable artifact
on every seed, while warm search used 30,208 target verifier bits versus
84,992 fresh bits: a 2.8136 transfer ratio, with a strict warm advantage on all
32 seeds. Source retention remained perfect; irrelevant inheritance,
destroyed reward, shuffled actions, and missing history admitted nothing.
Controller/operator updates and replay were zero. This establishes reusable
program structure and smallest-failed-binding relearning, not autonomous
controller-generated candidate libraries or physical transfer. The source and
target are now admitted together into a checksummed `.bank`; each artifact
contains the complete instruction stream and an allow-listed operator manifest.
Reload reconstructs fresh production operators, preserves artifact/bank
digests, and retains perfect source and target behavior. Arbitrary operator
imports, file corruption, controller-digest mismatch, rejected candidates, and
ambiguous multi-stream reads fail closed. The canonical `ExternalAgentBrainBank`
now stores this executive family beside the older temporal-address family in
one versioned JSON `.bank`; legacy torch banks require an explicit validated
migration, and their opaque route evidence is preserved rather than flattened.
The same bank can now compose an admitted receive-only fragment with an
admitted persistent temporal loop, record the parent slots and content
digests, and re-run the composed child after reload with perfect source
mastery. The composition gate is verifier-only and does not update the frozen
controller. Parent selection is now also memory-side: an opaque deterministic
ordered-pair search executes candidates on fresh verifier rollouts, appends the
first stable child, and records its unique bits-to-threshold, lifetimes, and
replay. Evaluation is staged: clearly sub-threshold first rollouts reject
immediately, while promising candidates receive fresh confirmation. A generic
control-flow reachability gate rejects non-final parents that
cannot hand off, preventing a persistent first loop from shadowing later
components. Persisted provenance is rebound by recomposing the recorded parents
and validating the admission receipt against the derived child.

The retained Brain Workshop evidence qualifies bounded append-only external
working-memory computation through n-back-32 with frozen source retention and
zero replay. The live diagnostic additionally establishes immediate batch-one
acquisition from clean-room RGB and waveform devices, including exact
source-preserving dual-stream behavior. It does not yet qualify physical
desktop mastery, autonomous general program induction, cross-mechanism
maintenance-policy transfer, unrestricted memory growth, or a complete
interactive executive ISA.

The physical Position 1-Back rung qualifies the narrower live-I/O claim: public
display capture, spatial onset segmentation, ordinary keypress output, exact
receipt matching, checkpoint/resume, and immediate evidence-bound updates work
together in the real GUI. Controller relation, source-conditioning, and
intention-decoding weights are pretrained across variable frontend projections
and frozen for task acquisition. A fresh task file starts with a uniform
categorical temporal address and updates only from public verifier outcomes.
The promoted 2-cell campaign produced 86 external-program updates from 86
unique public outcomes, 0 controller updates, and 0 replay. It reached a stable
rolling-44 threshold at bit 44 and finished at `1.0000` over the final 44
outcomes. Normal learning passed 32/32 rendered seeds while frozen,
reward-shuffled, action-reversed, and missing-history controls passed 0/32.
This establishes bounded two-cell Position 1-Back acquisition and retention,
not larger grids, general program induction, or dual-stream operation; system
audio parity is still missing.

The production bank boundary and its physical-campaign handoff are implemented,
including stable admission, duplicate reuse, opaque reward routing, frozen
activation, persistence, and corruption rejection. A fresh six-lifetime
two-cell campaign then admitted one immutable program from 37 public outcomes.
Four independent read-only sessions selected it after three newly captured
learned stimulus events and produced 13/15 positive outcomes. Controller and
program updates were zero; the external route ledger alone received 15 scalar
observations. This promotes the bounded single-program lifecycle, not general
program induction. Distinguishing multiple rules additionally requires a
human-visible rule cue encoded through the normal frontend rather than a hidden
rule identifier.

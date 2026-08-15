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
of every public frame used to derive it. The physical Neural Workshop adapter
accepts only explicit visible feedback colors over a complete trial window.
Under default Neural Workshop scoring, green is positive, red/blue is negative,
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
The internal `HANDOFF` instruction advances across a composed component in the
same tick without emitting an intention; it is used only by the explicit
`final_emit_only` composition policy.
For throughput, the interpreter also exposes an internal owner-bound sealed
state lease. It performs the full tensor/state validation once, then retains
cheap structural checks and validates every event collection and operator
result on subsequent ticks. The lease is deliberately non-serializable and
cannot cross executive instances; the public `tick()` path remains defensive
for restored or externally supplied state.
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

The retained Neural Workshop evidence qualifies bounded append-only external
working-memory computation through n-back-32 with frozen source retention and
zero replay. The live diagnostic additionally establishes immediate batch-one
acquisition from clean-room RGB and waveform devices, including exact
source-preserving dual-stream behavior. It does not yet qualify physical
desktop mastery, autonomous general program induction, cross-mechanism
maintenance-policy transfer, unrestricted memory growth, or a complete
interactive executive ISA.

An admitted executive artifact can now be saved and reloaded through the
canonical `AgentBrain.bank`, then run inside `CognitiveTickRuntime` through an
`ExternalExecutiveLiveMachine`. In a bounded frozen-controller diagnostic, the
same generic temporal-equality artifact reached 1.0000 accuracy on 60-step
1-back and 2-back lifetimes after reload, with 59 and 58 unique eligible
verifier bits respectively and zero controller or executive-program updates.
Across three replicates, machine p50 was 0.85 ms / 0.83 ms and p99 was
2.64 ms / 1.63 ms. This qualifies durable bank-backed live execution and
WAIT-safe causal pacing; it does not qualify autonomous skill selection or
learning the skill itself.

The next bounded live rung now routes between two such reloaded artifacts with
`ExternalExecutiveRouterLiveMachine`. A replaceable context encoder consumes the
first visible learned cue event, and the memory-side route ledger records one
mean verifier outcome per completed lifetime by default. Sixteen alternating
1-back/2-back training lifetimes (136 unique verifier bits) learned cue-specific
slot preferences with zero controller, decoder, executive-program, or replay
updates; three held-out lifetimes per route then scored 1.0000 on every 9-bit
1-back and 8-bit 2-back lifetime.
The evidence is intentionally bounded: the cue is public and ordinary, the
skills are pre-verified bank artifacts, and the route adapter is not yet an
autonomous context encoder. Per-action outcome transport remains available, but
it is not used as the default route mastery signal because partial lucky
streaks can otherwise promote the wrong slot.

The route policy has also passed a bounded nonstationary audit. After an exact
`.bank` reload, the private verifier rule changed from 1-back to 2-back behind
the same learned cue. Two failing lifetimes demoted slot 0; the policy avoided
that recently reversed slot, probed slot 1, and retained it across three perfect
lifetimes. Forced slot-0 retention remained perfect, and a cue-shuffled control
fell below threshold. This is causal route reversal and immutable-skill
retention, not autonomous program induction or a semantic rule switch.

The live route also now uses the same nearest-protected-context policy for
behavior probabilities as for preferred-order queries. After reload, six
previously unseen perturbed learned-event keys selected the correct frozen
skill and scored 1.0000 on 51 eligible verifier bits; each variant became an
independent row only after its first outcome. This is bounded representation-
radius generalization, not a hand-written semantic map.

The next live rung now grows a skill from the bank itself. A deterministic
parent-slot proposal derived a finite learned-event prelude and an admitted
delay-2 loop into a child artifact. Explicit compatible-operator sharing carries
stateful interfaces across the boundary, while `final_emit_only` rewrites
intermediate `EMIT` instructions to internal `HANDOFF`s so the same learned
event cannot create an extra external action. The child scored 1.0000 on 24
fresh verifier bits across three live lifetimes, entered slot 2 with parent
digests and composition policy persisted, and retained 1.0000 after exact bank
reload. A matched delay-1 parent composition scored 0.50, 0.375, and 0.50 and
was rejected without changing its bank. Controller, decoder, program, and
replay updates were zero. This promotes bounded bank-fed composition and event
continuity, not autonomous open-ended program synthesis or physical desktop
deployment.

The physical Position 1-Back rung qualifies the narrower live-I/O claim
against Neural Workshop's public window, not a second game:
public display capture, spatial onset segmentation, ordinary keypress
output, exact receipt matching, checkpoint/resume, and immediate
evidence-bound updates work together in that GUI. Controller relation, source-conditioning, and
intention-decoding weights are pretrained across variable frontend projections
and frozen for task acquisition. A fresh task file starts with a uniform
categorical temporal address and updates only from public verifier outcomes.
The promoted 2-cell campaign produced 86 external-program updates from 86
unique public outcomes, 0 controller updates, and 0 replay. It reached a stable
rolling-44 threshold at bit 44 and finished at `1.0000` over the final 44
outcomes. Normal learning passed 32/32 rendered seeds while frozen,
reward-shuffled, action-reversed, and missing-history controls passed 0/32.
This establishes bounded two-cell Position 1-Back acquisition and retention,
not larger grids, general program induction, or Dual acquisition from a blank
file. Desktop Dual now has a ScreenCaptureKit window tap that publishes
public PCM; missing or silent audio fails closed. That is an I/O path, not
a measured desktop Dual lifetime.

A separate Neural Workshop Dual path now publishes the queued stimulus
waveform on the public observation, never a letter ID. A replaceable audio
encoder emits a second amodal event; the frozen two-way decoder packs one
match bit per bound source onto the two public ports. Privileged keys fail
closed. Frozen `PREVIOUS` composition scored `1.000` on Dual 1-back and
2-back across two 60-trial seeds, with a wrong-depth control near `0.10`.
This qualifies Dual as a learner-visible I/O path. A blank address file now
also acquires Dual 1-back on both the rendered device and Neural Workshop
public PCM, retains it, and executes a one-step composition on Dual 2-back.
Mixed Dual labels are packed exact-match credit, not half-credit. This does
not qualify a measured desktop Dual lifetime or autonomous program induction.
A later unused three-seed holdout promoted header transfer and first-time
depth invention on Neural Workshop: same-slot 3-cell 3-back retrieve was
`3.88×` cheaper than a matched climb, and first-time 2-cell 3-back compose
was `1.64×` cheaper, with wrong-depth, missing-history, and reversal
controls below threshold. That lease is consumed. It does not claim
open program search or desktop Dual.

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

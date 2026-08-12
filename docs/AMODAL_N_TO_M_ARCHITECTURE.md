# Canonical architecture: amodal N-to-M neural computer

This document is the normative architecture specification for the project.
When a design note, experiment README, or historical report uses looser
language, this document controls the meaning of the target system.

The continual-learning specialization of this architecture is defined in
`docs/POLICY_FREE_CONTINUAL_LEARNING.md`: durable general knowledge is factual
transition structure, while behavior is derived by opaque model search rather
than stored as a task policy.

## Evidence provenance

References to the exported games session are architectural provenance and
hypothesis sources, not independent evidence. Any quantitative or causal
claim originating only in that export is **SINGLE-SOURCE, UNREPLICATED** until
an in-repository rerun satisfies the experiment ladder and controls. A claim
with repository evidence must point to its own `session_records/` report and
must be scoped to that report's measured task family and controls.

## One-sentence definition

The target is one modality-independent cognitive controller that consumes a
variable number **N** of simultaneous or asynchronous encoded event streams and
emits a variable number **M** of abstract intention streams, which independent
decoders or actuators translate into concrete outputs.

## What “one model” means

The project requires **one sovereign cognitive model**: the AI controller. It
does not require every learned component to occupy one monolithic checkpoint.
An encoder or decoder may itself be a neural model—including a fine-tuned
language model—but it is an adapter, not a second reasoner.

The boundary is functional:

- the controller performs modality-independent memory, inference, planning,
  learning, and intention formation;
- encoders translate raw streams into the shared neural interface;
- decoders translate intentions into output formats or physical effects;
- adapters must not solve tasks independently, retain a parallel world model,
  or bypass controller state and intentions;
- replacing an adapter must not require replacing or branching the controller.

Thus “one controller + N encoders + M decoders” is compatible with multiple
neural modules while preserving one cognitive agent.

```text
raw input streams                    amodal core                    raw outputs

camera 1 -> vision encoder 1 --\                              /-> text decoder
camera 2 -> vision encoder 2 ---\                            /--> speech decoder
audio    -> audio encoder -------+-> amodal event bus -> one +---> image decoder
text     -> language encoder ----+   AI controller       +---> robot actuator
touch    -> touch encoder -------+   RAM/VRAM + disk     +---> JSON adapter
sensor N -> encoder N -----------/                            \--> decoder M
```

Both **N** and **M** vary at runtime. Adding a sensor or output device must not
change the controller's parameter shapes or require a new controller branch.
Finite hardware still imposes bandwidth, latency, and memory limits; “no fixed
architectural limit” does not mean infinite simultaneous compute.

## LLVM analogy

```text
source language -> compiler frontend -> LLVM IR -> machine backend -> code
raw modality    -> learned encoder    -> neural IR -> decoder/actuator -> output
```

The analogy defines the modular boundary:

- Encoders are learned frontends. They lower modality-specific data into the
  shared neural event interface.
- The controller and its memory operate only on the neural intermediate
  representation. They do not parse pixels, waveforms, tokens, JSON, or device
  commands.
- Decoders and actuators are learned or calibrated backends. They lower
  abstract intentions into modality- or device-specific outputs.

Unlike LLVM IR, the semantic content of the neural IR is not hand-written.
Tensor shapes and transport rules are engineered; latent meaning must emerge
because it improves verified behavior, transfer, compression, and learning
speed.

## Terminology and claim levels

Use these terms consistently:

- **raw stream**: pixels, waveform samples, text tokens, or sensor readings;
- **encoder/frontend**: modality-specific translator from a raw stream;
- **neural IR / amodal event**: opaque learned transport representation shared
  across frontends;
- **controller**: the sole general learner and reasoner;
- **working memory**: fast, mutable RAM/VRAM state used for current computation;
- **long-term memory**: persistent disk-backed learned experience;
- **intention**: opaque controller output describing intended content or effect;
- **decoder/backend/actuator**: translator from intention to an external format
  or effect;
- **whole agent**: adapters, controller, memory hierarchy, and runtime buses.

Claims have three levels and must not be conflated:

1. **Integrated prototype:** modality submodules exist but are owned by one
   checkpoint and the controller API still accepts raw input.
2. **Modular neural IR:** adapters are independently owned and replaceable, and
   behavior survives the extracted interface.
3. **Audited amodal N-to-M system:** variable-cardinality simultaneous inputs
   and outputs pass replacement, composition, no-bypass, and causal audits.

An opaque latent is not automatically an amodal concept. “Amodal” is earned by
cross-adapter behavioral reuse and causal tests, not by naming a vector.

## Architectural components

### 1. Input adapters: zero to N encoders

Each connected input stream has an encoder appropriate to its raw format. A
bidirectional modality such as language normally has both an encoder and a
decoder; a camera may be input-only and a keyboard actuator output-only.

Examples:

- image/video -> vision encoder;
- waveform -> audio encoder;
- text/token stream -> fine-tuned language encoder;
- touch, proprioception, temperature, radar, or future sensor -> its encoder.

Encoders may be pretrained and then fine-tuned as neural-IR frontends. A normal
LLM is not automatically a language adapter: its hidden states require a
trained bridge into the shared event representation.

Encoders decide when to emit events. A vision encoder may emit on scene change;
an audio encoder may suppress silence; a transcript encoder may emit per phrase.
No global fixed sensor rate is required.

### 2. Amodal event bus

All encoders emit the same generic transport type. A conceptual interface is:

```python
@dataclass
class AmodalEvent:
    payload: Tensor[event_width]       # learned opaque content
    source_key: Tensor[source_width]   # learned adapter/source identity
    timestamp: Tensor                  # generic temporal coordinate
    duration: Tensor                   # optional temporal extent
    confidence: Tensor                 # generic presence/reliability signal
```

The exact schema may evolve, but it must satisfy these invariants:

- `payload` width is independent of raw modality and sensor count;
- payload coordinates have no hand-assigned human semantics;
- a cycle accepts a variable-size set or sequence of events;
- events from different streams can be simultaneous, delayed, missing,
  redundant, contradictory, or differently sampled;
- source identity is a learned generic key, not a task label or semantic
  instruction such as `vision_is_authoritative`;
- new adapters register without resizing the controller.

Approximately simultaneous events should remain separately addressable. They
must not be blindly averaged. A lightweight set/attention stage may bind a
small temporal window before or during recurrent controller updates; ordering
between windows remains causal and timestamp-aware.

### 3. One modality-independent AI controller

The controller receives `AmodalEvent` objects, its own prior state and actions,
memory reads, and scalar verified outcomes. It must never receive raw pixels,
waveforms, text tokens, semantic task IDs, correct answers, or device protocols.

The controller owns the general cognitive machinery:

- recurrent state and event integration;
- cross-stream attention and binding;
- temporal ordering;
- RAM/VRAM working memory;
- content-addressed long-term-memory queries and writes;
- comparison, retrieval, transformation, planning, and confidence;
- learned compute, memory, and attention allocation;
- abstract intention formation.

The controller must learn which events are relevant, which modalities agree,
which sources are trustworthy, what should be retained, and what can be
discarded. Those policies must not be hard-coded per modality.

### 4. Memory hierarchy

Fast mutable state and active context live in RAM/VRAM. Persistent episodic and
procedural rows live on disk and can be retrieved into fast memory. Memory
contents use the same learned representational currency as the controller; they
do not store privileged game state or English reasoning traces.

Adapter replacement must not invalidate unrelated long-term memories. Cross-
modal qualification should eventually show that a memory written through one
encoder can be retrieved and used when queried through another.

The production memory boundary is `MemoryBackend` v1. `MemoryQuery`,
`MemoryRead`, and `MemoryWriteReceipt` are versioned learned-tensor records;
they carry no modality, protocol, task, or privileged-state fields.
`ContentAddressedMemory` is the bounded in-process implementation and
`PersistentContentAddressedMemory` is its disk-backed replacement. Matching
keys are upserted using a versioned cosine-similarity threshold, while
unmatched rows use bounded strength-based eviction. Persistent
writes use an atomic, checksummed snapshot format
(`neural-computer.memory-snapshot.v2`; v1 snapshots remain readable),
validate state before and after loading, and roll back an in-memory mutation if
the durable snapshot fails. Runtime checkpoint loads use the same post-load
state validation and rollback for memory components. The contract makes
replacement possible, but it does not yet claim crash-consistent multi-process
access, cross-adapter retrieval gains, or general learned memory utility.
`MemoryCandidates` and the generic `target_index` write request now permit an
independently versioned memory-side eviction policy without exposing physical
row identity to the controller. A paired outcome-only audit promotes a narrow
three-slot/two-row learned-eviction result with the controller frozen,
fresh-token parent acquisition, persistent reload, checksum rejection, and a
reward-shuffled chance control. This qualifies learned utility for the audited
bounded eviction problem only; general episodic utility and unrestricted
memory growth remain open.
The follow-on `AppendOnlyContentAddressedMemory` boundary removes the fixed
row-count assumption for storage itself: unmatched learned keys append to a
variable-length, checksummed state, while matching keys upsert in place. A
frozen-controller audit at 64, 256, and 1,024 opaque records replicates
permuted exact recall, zero clear-memory hits, and persistent reload/recovery
across two seeds. This promotes logically growing external storage, not
learned compression, new procedure acquisition, or general continual
learning; the latter remain the next frontier.

The canonical memory boundary now also owns a persistent
`CapabilityRetentionLedger`. It observes only opaque learned keys and scalar
verifier outcomes, promotes a capability only when its cumulative mastery
threshold remains satisfied at every measured prefix, and masks protected
rows from the default strength fallback as well as executable-artifact
eviction. Sustained low outcomes use hysteresis before declaring a reversal;
when all rows are protected, the write fails explicitly so the caller must
grow or transactionally consolidate the bank. `evaluate_retention_gate`
requires a new capability's stable prefix and the complete retained-score
floor before a consolidation can be adopted. The ledger is persisted beside
 disk memory, runtime checkpoints, and artifact compaction. This makes the
 retention boundary real and replay-free. The generated length-six Brain
 Workshop audit now promotes new-skill acquisition, reversal, full-bank
 protection, and retention together; unrestricted learned utility and general
 continual learning remain open.
Memory reads preserve gradients through query-key scoring and value weighting;
inside an explicit differentiable transaction, pending values are mixed by a
trainable write-strength gate. Durable storage mutation remains detached and
explicitly stateful so persistence never captures an autograd graph.
The canonical executable-artifact store additionally exposes verified top-k
opaque candidate promotion. This permits a caller to measure or compose
reusable learned factors without adding task semantics to the memory backend;
single-artifact execution, multi-artifact composition, and transfer policy
remain caller-owned and independently replaceable.
Its consolidation transaction can build a disposable candidate before running
the retention gate through a caller-supplied opaque outcome probe. This keeps
protected-source consolidation honest: candidate behavior is measured after
construction, the source remains immutable, and adoption still requires both
stable replacement mastery and retained-capability evidence.
The follow-on executable-artifact audit trains an opaque permutation-equivariant
address router over the compacted views. The selected candidate is resolved
through the generic memory promotion path before execution, and route,
permutation, reward-shuffle, wrong-view, reload, and frozen-core controls pass
on two seeds. This promotes bounded learned address acquisition, not arbitrary
new skill induction or unrestricted continual learning.
The production package now also exposes the caller-owned
`compose_growth_artifacts()` merge: it remaps verified payloads into disjoint
growth namespaces and rejects collisions before generic frozen-core loading.
A paired working-memory audit promotes the narrow result that two such factors
can execute together in one frozen controller while retaining both private
procedures; sequential factor algebra and arbitrary program synthesis remain
unqualified.
The follow-on working-memory audit now qualifies a narrower sequential ABI:
an external producer artifact feeds a prior-only recurrent consumer slot, and
the pair learns span-twelve global parity while each isolated factor remains at
chance. This is evidence for learned register-to-register composition under a
frozen core, not unrestricted program algebra. The generic producer/consumer
growth-register boundary is now implemented in the canonical controller and
passes a two-seed rendered-event pressure test; the older span-twelve result
remains an archived compatibility-fixture result.
An optional opaque runtime memory scope selects an isolated fixed-capacity bank
for each independent batched trajectory; it is execution context rather than a
semantic, modality, or task field, and the default single-scope layout remains
checkpoint-compatible.
A narrow outcome-only scalar-recall rung is promoted under
`session_records/memory_recall_amodal_2026-08-04/`: three seeds reproduce a
scalar verifier outcome after recurrent-state reset, clear-memory and
corruption controls remain at chance, and a persistent backend replacement
retains perfect recall. The rung uses a fixed `0.5` commit threshold and an
explicit differentiable write/read transaction. Ordinary commit rates were
1.0, 0.6563, and 1.0; this shows the gate can affect writes but does not
qualify learned skip policy or utility-based retention alongside multi-row
interference handling and cross-adapter retrieval.

### 5. Amodal intention bus: zero to M outputs

The controller emits a variable sequence or set of opaque `IntentEvent`
objects. It does not emit keyboard codes, JSON fields, words, audio samples, or
robot voltages directly.

```python
@dataclass
class IntentEvent:
    payload: Tensor[intention_width]   # learned abstract intended content/effect
    timestamp: Tensor
    confidence: Tensor
    target_key: Tensor | None          # learned routing/subscription key
```

Any number of compatible output adapters may subscribe to the same intentions.
One intention may simultaneously drive text, speech, a display, and a robot.
The controller must not need to know the output format or how many decoders are
connected.

The canonical package now includes a keyboard instance of this boundary.
`KeypressDecoder` consumes the learned `IntentEvent` payload and returns
logits over an externally owned key-index space; its `decide()` method can
sample a key and records the exact selected-action propensity. The matching
`KeypressEncoder` turns a previously logged external key index into the opaque
feedback embedding carried by `ControllerFeedback`. Key indices and keyboard
layout remain outside the controller, so this adapter is replaceable and does
not create a keypress-specific reasoning path.

For example, an abstract answer grounded in the visual concept blue may be
lowered into:

- `"The color is blue."` by a fine-tuned language decoder;
- spoken audio by a speech decoder;
- `{"color": "blue"}` by a protocol adapter;
- a blue swatch by a display adapter;
- a pointing action by a calibrated robot actuator.

These are multiple realizations of one learned intention, not separate
controller concepts hand-coded as `BLUE`.

## Required modularity properties

The architecture is not considered amodal merely because modules have separate
Python classes. The following behavioral properties are required:

1. **Frontend replacement:** a new encoder can expose existing controller
   skills without retraining the cognitive core.
2. **Backend replacement:** a new decoder can express existing intentions
   without retraining the cognitive core.
3. **N-input composition:** a task can require evidence split across several
   encoders, where no single stream suffices.
4. **M-output fan-out:** one frozen intention can be rendered correctly through
   several independently qualified outputs at once.
5. **Variable cardinality:** sensor and decoder counts change at runtime without
   controller reconstruction.
6. **Asynchrony:** input rates, delays, and absences vary without fixed slots or
   fixed modality order.
7. **No bypass:** shuffling or removing controller events/intentions must break
   performance even when an attached encoder or decoder is powerful.
8. **Stable neural IR:** qualifying a new adapter must not silently redefine the
   interface and break already-qualified adapters or memories.

## Language example

```text
text:  "What color is the square?" -> trained language encoder --\
                                                               +-> controller
image: blue square               -> trained vision encoder ----/
                                                                    |
                                                         abstract answer intent
                                                                    |
                                              trained language decoder/LLM
                                                                    |
                                                     "The color is blue."
```

The language model is a frontend/backend compiler for the neural IR, not the
sovereign reasoner. It must be fine-tuned or adapted in both directions. The
image must change the answer causally; the decoder must fail when the
controller intention is shuffled or removed.

## Training and audit requirements

Adapters may be trained with procedural environments and deterministic
verifiers, but the deployed learner receives only encoded sensory experience,
its own opaque actions, its memory/state, and scalar outcomes. Diagnostic
semantic probes are discarded and never enter the deployed system.

Every adapter or multimodal claim should include:

- encoder information-preservation probes;
- decoder fidelity tests on frozen intentions;
- aligned versus time-shuffled streams;
- missing, delayed, duplicated, noisy, and contradictory sources;
- modality dropout and source-count variation;
- controller-event and controller-intention shuffle/bypass controls;
- unseen encoder and decoder swaps with the controller frozen;
- comparisons of stable verifier bits-to-threshold against fresh learning;
- retention audits on the complete admitted repertoire.

The key compounding metric is whether training one new adapter unlocks many
existing cognitive skills for less experience than relearning those skills in
the new modality.

### Zero-semantic-label rule

The deployed learner is not trained with human-authored concept names, semantic
slot labels, task IDs, correct unattempted actions, or solution traces. It may
learn from raw/encoded experience, its own attempted actions, self-supervised
prediction, and scalar outcomes from deterministic verifiers. Semantic labels
may be used only by discarded diagnostic probes and never transferred into the
deployed weights.

Any use of pretrained adapters must be disclosed separately. Their inherited
knowledge does not count as zero-label capability learned by this project, and
the no-bypass audits must show that they cannot solve the benchmark without the
controller.

## Current implementation boundary (August 2026)

The canonical production implementation is now under `src/neural_computer/`.
It exposes an opaque event collection, an event-token controller, generic
feedback vectors, and an intention fan-out runtime. The older integrated
controller and its checkpoint converters remain under
`experiments/archive/unified_cognitive_controller/` solely for historical replay and
are not the production agent API.

The production package closes the structural boundary gaps as follows:

- `EventTokenWindow` retains payloads, presence, confidence, source keys,
  timestamps, durations, and age across controller updates; attention is an
  explicit per-token operation rather than a pre-controller reducer. The
  controller also has a zero-initialized bounded pairwise event-attention
  residual for learned cross-token binding; it adds no modality-specific
  branch and preserves the prior until trained. The v9 controller additionally
  exposes a zero-initialized bilinear interaction between prior opaque
  feedback, generic source keys, and current event tokens so learned outcome
  context can update evidence binding without a protocol-specific branch.
- `ControllerFeedback` carries an opaque learned feedback vector plus scalar
  reward, propensity, and feedback presence. No discrete device-action space
  enters the controller.
- `EpisodicContextEncoder.step()` provides an online, replaceable recurrent
  context state from learned events, opaque actions, scalar outcomes, and
  presence. `EpisodicIntentAdapter` can add a zero-initialized external
  context residual to an opaque intention before decoder fan-out; neither
  component adds task IDs or a protocol-specific controller branch.
- `MemoryBackend` v1 is the replaceable runtime contract for controller
  queries, reads, writes, receipts, and configuration. The in-process
  `ContentAddressedMemory` and disk-backed
  `PersistentContentAddressedMemory` implement the same boundary; snapshots
  have their own explicit version and validated state schema.
- Generic reliability and wait policies can be trained from latent/transport
  evidence and scalar utility without modality-specific branches.
- `ExternalOnlineStreamBindingMemory` is the memory-side learned identity
  boundary for transition streams. It consumes one learned transition arrival
  at a time, maintains anonymous bounded prefixes, and returns an opaque stable
  source key only when similarity and separation are sufficient. Delay and
  reliability are external sufficient state; unresolved binding is an
  explicit non-mutating outcome. `ExternalLearnedMultiStreamTransitionContextRouter`
  composes this binder with one shared factual transition bank, removing
  caller-owned stream keys without adding a controller branch.
- The recurrent controller emits an execution-plane `WAIT`/`THINK`/`COMMIT`
  policy. `WAIT` preserves a tentative state for later events, `THINK` spends
  a bounded quiet recurrent tick, and `COMMIT` releases the opaque intention;
  this is scheduler control inside the same model, not a second reasoner.
- The execution head receives a small generic transport summary—event density,
  aggregate confidence, and their interaction—rather than a lossy scalar
  availability signal. These features remain independent of modality,
  protocol, task identity, and semantic labels.
- `save_runtime` and `load_runtime_components` serialize controller, encoders,
  decoders, memory, and transport policies as independently loadable
  components.

These are structural and causal-contract milestones; they do not by themselves
qualify natural-language grounding or broad multimodal transfer.

Protected plasticity is trainer infrastructure rather than a deployed
reasoning module. The canonical package exposes aggregate rehearsal-gradient
projection and detached gradient accumulation so a new update can remove only
the component that conflicts with verified old behavior. It is compatible with
zero-impact growth adapters and preserves the learner-visible boundary: the
controller still receives only learned events, opaque attempted actions, and
scalar verifier outcomes. Any use of the mechanism must retain fresh old-skill
audits, shuffled-outcome controls, and unique-experience accounting.

## Composable fragment bank boundary (2026-08-11)

The durable unit of external growth is now defined as a **reusable fragment**,
not a task-sized program. This distinction is normative:

- the bank owns one shared opaque operator basis;
- a fragment stores only a short coefficient sequence over that basis and an
  opaque address key;
- a composition query retrieves fragments by learned event/intention evidence;
- the shared register interpreter executes the resulting ordered chain;
- the controller and all output adapters keep their parameter shapes;
- fragment rows may grow, persist, be protected, or be replaced outside the
  controller.

The shared basis is itself append-expandable external state. When a new basis
prefix is appended, every existing coefficient row receives zero padding, so
old fragment codes are unchanged at the growth boundary. A mastered basis
prefix can be gradient-protected while a candidate trains only new directions
and its own coefficients. This is the escape hatch from a saturated fixed
basis; it does not resize the controller or claim that new directions already
implement useful computation.

There must be no architectural concept called `snake_program`,
`pong_program`, or an equivalent task-indexed skill branch. A game, task, or
modality may be the verifier's source of experience, but it is not a storage
schema. Repeated structure earns reuse only when independently acquired
fragments are selected and composed by fresh opaque evidence. Parent digests
are provenance and checksums, not semantic labels.

The production seam is `ExternalSkillFragmentBank` plus
`ExternalCapabilityRegisterMachine.execute_fragment_composition()`. The bank
routes with cosine address evidence and an optional outcome-trained,
permutation-equivariant residual scorer. The returned composition is padded
only at the transport boundary; the interpreter removes padding before
execution, so variable fragment lengths do not become learned no-op
instructions. Composition is therefore serial and order-sensitive by design;
any future commutative or simultaneous composition operator needs its own
causal audit.

This design combines the strongest lessons from the architecture review:

1. **Shared basis before task artifacts.** Independent full-task modules tend
   to coexist without discovering common factors. A common basis gives later
   fragments a place to reuse operators while retaining independently
   addressable external coefficients. If the basis is saturated, append
   protected directions rather than widening the controller.
2. **Trained routing before arithmetic.** Raw summation of opaque vectors is
   compact but interference-prone. The safe default is content-addressed
   selection and serial execution; learned residual routing may be enabled only
   after fresh scalar-outcome evidence. Any weighted blending or discrete
   multi-fragment search must be measured separately.
3. **Fast working state and durable fragments stay separate.** The fragment
   bank is long-term external state; the register is a transient execution
   snapshot. Observed evidence must not be overwritten by execution results,
   and downstream fragments must not reread raw events.
4. **Composition is not arbitrary new computation.** Fragments can select and
   recombine computation the shared interpreter already expresses. New
   algorithmic depth still requires verified training of the shared meta-
   machinery or a separately admitted external compute basis. Theoretical
   universality is an expressibility target, not a current capability claim.

The current unit tests establish only the structural boundary: shared-basis
materialization, route permutation equivariance, outcome-only route gradients,
variable-length execution, append-without-resize, persistence, and checksum
validation. They do **not** promote positive transfer, arbitrary program
induction, unrestricted memory growth, or general continual learning. Those
claims require fresh rendered-event experiments with stable-prefix mastery,
wrong-fragment and blank-bank controls, route permutation, reward-shuffled,
missing-evidence, memory-corruption, fresh-learner, and zero-replay accounting.

### Replicated trace-combiner result (2026-08-11)

The first implementation that made the boundary causally useful exposed a
missing computational seam: executing a chain and returning only its final
register state gave the downstream decoder no learned way to distinguish the
ordered intermediate transformations. The canonical external path now has
three stages:

```text
opaque event/intention query
        -> external fragment bank
        -> shared register interpreter trace
        -> external trace combiner
        -> intention/output decoder
```

`ExternalSkillFragmentExecutionTrace` preserves ordered post-instruction states
and padding masks, while `ExternalSkillFragmentCombiner` reads only that trace
and remains outside the frozen controller. This is still one external memory
and compute boundary: it does not introduce a task-specific reasoning branch,
raw-event shortcut, verifier metadata, or protocol-shaped controller input.
Normalized fragment-code materialization is part of the boundary contract. It
prevents independently learned coefficient rows and basis rows from producing
numerically negligible instructions solely because their product scale is small.

The fresh-rendered audit
`experiments/external_skill_fragment_composition_amodal/` passed on seeds 69316
and 69317 at 128 composition updates. Stable inherited-versus-fresh verifier
bits were `(6,144, 24,576)` and `(9,216, 12,288)`, respectively; every seed
passed old-fragment retention, reversed-order rejection, zero-code ablation,
reward-shuffled rejection, frozen-parent digest, no-replay, and route-resolution
gates. This promotes bounded reusable compositional transfer only. It does not
promote arbitrary new computation, unrestricted memory growth, compression, or
general continual learning. The next four-fragment closure audit is recorded
under `session_records/external_skill_fragment_multi_composition_amodal_2026-08-11/`.

### Four-fragment closure and acquisition isolation (2026-08-11)

The next pressure test passed that expansion on two matched seeds. Four
fragments—`reverse`, `rotate`, `complement`, and `prefix_parity`—were acquired
sequentially with fresh outcomes, then protected. A new external trace combiner
learned the held-out order
`prefix_parity -> complement -> reverse -> rotate` while the parent controller,
interpreter, and acquired bank were frozen. Stable inherited composition cost
6,144 verifier bits on both seeds; matched fresh learners cost 12,288 bits on
both seeds. All four primitive stable-mastery and post-composition retention
gates passed, as did reversed-order, zero-code, missing-evidence,
reward-shuffled, frozen-parent, persistence-corruption, route-resolution, and
zero-replay controls.

This audit also resolved a training-design error. A preliminary rung used the
same decoder loss to acquire a primitive and a longer composition involving
that primitive; one seed then failed primitive retention despite passing the
composition score. The canonical acquisition rule is now: primitive objective
alone -> stable-prefix gate -> protect fragment -> composition objective with a
new external combiner. This is a general continual-learning principle for the
architecture: do not ask one external memory cell to be both an atomic skill
and a composite program during the same credit-assignment window.

The result promotes bounded four-fragment reusable composition, not arbitrary
program induction, unrestricted memory growth, compression, or general
continual learning.

### Multi-target frozen-bank closure (2026-08-11)

The next audit reused the same acquired four-fragment bank across three
independently held-out orders. Each order received a new external trace
combiner and output decoder; the acquired register interpreter and fragment
bank were frozen and checksummed before and after target learning. On two
matched seeds, all three inherited targets reached stable mastery at 6,144
verifier bits, while matched fresh learners required 12,288 bits for every
target. Final inherited target accuracy ranged from 0.917 to 1.000 across the
two seeds. Reversed-order, zero-code, missing-evidence, and reward-shuffled
controls were rejected for every target, with zero replayed examples.

The first 64-update replication is retained as a useful rejected diagnostic:
one matched fresh target ended at 0.75 and had no stable prefix. Doubling only
composition exposure resolved the failure at 128 updates without changing the
boundary or relaxing a causal gate. That result identifies fresh target
optimization length/variance—not bank mutation—as the current bottleneck for
scaling this mechanism.

This promotes reusable bounded continual-memory composition across multiple
programs. It still does not establish arbitrary program induction, unrestricted
memory growth, compression, or general continual learning. The next pressure
test should hold the acquired bank fixed while growing the number of target
programs and varying composition depth, then test whether a shared external
learner can amortize acquisition without turning target-specific combiners into
an unbounded collection of adapters.

### Shared-composition pressure test and learner-view isolation (2026-08-11)

The next implementation makes the richer execution boundary explicit. A
versioned trace may carry learned instruction codes, transition deltas, and
opaque fragment segment lengths, which preserves operator evidence and file
boundaries without exposing raw events or verifier metadata. Routing receipts
remain on the memory side: `ExternalSkillFragmentLearnerTrace` is a separate
learner ABI that contains no fragment indices, route scores, or bank
cardinality. This makes the no-address-shortcut rule enforceable rather than a
property of one combiner implementation. Variable-length rows are grouped by
executable length in the register transport path, preserving exact semantics
while avoiding one interpreter call per batch row.

One shared segment-aware combiner and decoder were then trained across three
opaque orders with three held-out orders, while the parent and acquired bank
were frozen. The shared learner reached `0.6536/0.9531/0.7760` on training
orders and `0.6276/0.5182/0.6094` on held-out orders; it did not reach a stable
prefix. Wrong-order, zero-code, missing-evidence, reward-shuffled,
frozen-digest, persistence/corruption, and zero-replay controls passed, but the
capability gate is rejected. The batched transport reduced this audit from
496.5 to 353.5 seconds, which is an implementation gain rather than a learned
capability gain. State-only traces, rich flat traces, an atomic-anchor loss,
and a six-order/64-update coverage rung were also rejected; their durable
decision record is `session_records/external_skill_fragment_shared_multi_target_v2_2026-08-11/`.

The result sharpens the bottleneck: the acquired files are isolated and
reusable, but one learner has not yet inferred a general composition law from
the available order coverage. The next rung must vary composition depth and
provide a curriculum of fresh opaque orders to the same shared learner, with
stable-prefix and fresh-learner accounting. Allocating one new combiner per
target would conceal this limitation and is not the canonical architecture.

### Shared operator-algebra diagnostic (2026-08-11)

The first attempt to address that bottleneck adds a replaceable external
`ExternalSkillFragmentOperatorCombiner`. It summarizes each rich learner-view
segment and applies one code-conditioned low-rank state transition to a shared
composition state; it has no fragment-index, verifier, or depth-specific input
and persists through a versioned checksum-validated memory file.

At the matched seed-69316 64/256/128 audit, shared training accuracy was
`0.6849/0.7266/0.7786` and held-out accuracy was `0.6016/0.5833/0.7083`.
Wrong-order accuracy remained `0.6563/0.6745/0.7214`, so the learner had not
acquired reliable ordered binding. Frozen-parent, frozen-bank, zero-code,
missing-evidence, reward-shuffled, persistence, and zero-replay controls passed,
but no stable prefix or capability promotion was reached. The durable decision
record is `session_records/external_skill_fragment_operator_algebra_rejected_2026-08-11/`.

The ABI and persistence seam are retained as infrastructure; the low-rank
operator is not promoted as a learned composition law. The immediate bottleneck
is now ordered credit assignment: the learner can fit some target behavior but
does not consistently distinguish a valid sequence from a cyclically shifted
one. The next experiment should use a smaller curriculum with explicit paired
order contrasts before increasing depth or memory capacity.

### Ordered credit-assignment contrast diagnostic (2026-08-11)

The follow-up diagnostic corrected two transport defects before measuring a
new mechanism: mixed-depth programs are grouped by executable length, and
composition IDs are contiguous per target so per-target audits are valid. It
then added an optional trainer-only counterfactual loss that reruns each
multi-step example with a cyclically shifted opaque route and inverted action
utilities. The contrast was applied equally to the shared, reward-shuffled,
and fresh arms.

At the matched seed-69316 16/64/64 audit, the baseline reached held-out order
accuracy `0.5313/0.6250/0.6354` (mean `0.5972`); the contrast reached
`0.5104/0.5729/0.4792` (mean `0.5208`). It improved wrong-order rejection
(`0.3542/0.5521/0.3125`) but consumed 576 paired counterfactual rollouts,
reached no stable prefix, and did not improve held-out transfer. Frozen-parent,
frozen-bank, no-bypass, missing-evidence, reward-shuffled, persistence, and
zero-replay controls passed. The decision record is
`session_records/external_skill_fragment_order_contrast_rejected_2026-08-11/`.

The contrast is therefore not promoted as a learned composition law. The
useful architectural result is narrower: route-level negative supervision can
teach rejection without teaching a reusable ordered execution law. The next
high-ROI rung should reuse frozen trace computation and expose operator-level
intermediate verifier signals or a protected step-indexed external execution
state, with stable-prefix and fresh-order controls unchanged.

The first variable-deliberation outcome-only rung is recorded under
`session_records/deliberation_amodal_2026-08-03/`. It validates the bounded
runtime path but is rejected as a learned capability promotion: the controller
retained its immediate-commit prior across three short seeds. The next rung
must improve execution-policy exploration and credit assignment before any
claim that the agent learned when to spend compute.

A follow-up run under
`session_records/deliberation_amodal_followup_2026-08-03/` now promotes a
narrower result: after observable transport warmup, the same controller learns
to `COMMIT` complete windows and `WAIT` for delayed partner evidence, with
replicated mixed-utility gains over immediate commit and fixed waiting.
Mixed-state `THINK` arbitration remains unqualified because the combined
distribution still collapses toward the globally safest action.

The isolated causal `THINK` primitive is now promoted under
`session_records/deliberation_think_required_2026-08-03/`: three seeds select
`THINK` for a low-confidence event whose partner is released only after the
quiet tick, while commit and wait controls remain near chance.

The mixed execution frontier is now promoted under
`session_records/deliberation_mixed_arbitration_2026-08-03/`. With the generic
transport summary and a balanced complete/delayed/think-required curriculum,
three outcome-only seeds select `COMMIT` on complete windows, `WAIT` on delayed
windows, and `THINK` when waiting cannot reveal the partner. Held-out mixed
reward is 1.0 for every seed and utility ranges from 0.858 to 0.866 against an
optimal 0.8625 under the recorded compute costs. This is a narrow verified execution
capability, not evidence of broad multimodal reasoning or language grounding.

That learned execution policy now also replays through the production
`AmodalEventWindowBuffer`, under
`session_records/deliberation_timestamp_buffer_2026-08-03/`. Three seeds
preserve `COMMIT` for out-of-order complete arrivals, `WAIT` for delayed
timestamp-jittered partners, and `THINK` for the bounded quiet-tick path, all
at reward 1.0. This promotes the integration of learned execution with the
timestamped event boundary. The follow-up timeout rung under
`session_records/deliberation_timeout_absence_2026-08-03/` then freezes the
promoted transport heads and trains only an age-gated timeout residual from
outcome-only missing-evidence episodes. On paired 1,024-episode mixed audits,
utility beats immediate commit by 0.0561, 0.0678, and 0.0551 for seeds 17, 18,
and 19; each seed commits after the bounded timeout when the partner is
permanently absent. This promotes a narrow bounded-termination primitive, not
general learned absence handling across modalities.

The canonical controller's source-reliability boundary is now promoted under
`session_records/reliability_amodal_2026-08-03/`. Four frozen raw frontends
render one high-bit event and three low-bit events with hidden source-specific
flip rates. The controller learns the generic reliability pattern from scalar
outcomes alone: across seeds 17, 18, and 19, a reliable source agreeing against
two noisy sources scores 0.9976, 0.9912, and 0.9966, while flipping that
reliable source scores 0.0015, 0.0103, and 0.0005. Stream-order, intervention,
and all-low-missing controls pass their pre-registered boundaries. This is a
narrow synthetic source-reliability result; learned cross-modal relevance and
general missing-stream inference remain open.

The next context-dependent relevance rung is now promoted under
`session_records/relevance_amodal_2026-08-03/`. The verifier randomizes which
of two candidate streams is relevant on every episode; only agreement between
the candidate's opaque content tag and a context event identifies it. Frozen
frontends and an exactly balanced hidden-assignment curriculum leave the
controller only scalar outcomes. Across seeds 17, 18, and 19, both forced
candidate assignments, candidate swaps, and stream-order permutations pass the
0.80 gate, while cross-episode candidate shuffling stays at 0.5093, 0.5142,
and 0.5137. This promotes synthetic context-dependent relevance through the
canonical event boundary, not arbitrary natural cross-modal grounding.

Context-conditioned contradiction arbitration is now also promoted under
`session_records/context_conflict_amodal_2026-08-03/`. Two candidate streams
always disagree, and a separate context stream privately determines whether B
or C is trustworthy. With frozen independent frontends and balanced scalar
outcome-only training, seeds 17, 18, and 19 reach 0.9995/1.0000, 1.0000, and
1.0000 clean reward respectively; both forced context values and stream-order
permutations pass, while assignment shuffling stays at 0.4956/0.5044/0.5068
and context inversion collapses to 0.0005/0.0000/0.0000. This is a narrow
stable context-to-source arbitration result; temporal trust reversal and
natural contradiction resolution remain open.

The historical outcome-only temporal source-trust calibration is recorded under
`session_records/calibration_conflict_amodal_2026-08-03/`. Two contradictory
streams contain a hidden but episode-stable reliable source; after an opaque
action and scalar verifier outcome, the controller must reuse that calibration
on later ticks. A generic learned source-credit policy updates persistent
source-key trust from prior event tokens and feedback. That record describes
the earlier v14 runtime and is a narrow historical result, not a qualification
of every later controller revision.
Seeds 17, 18, and 19 reach 1.0000/0.7552/1.0000 clean post-calibration reward
and 1.0000/0.7599/1.0000 under stream-order shuffling, while no-feedback,
feedback-shuffled, action-shuffled, and intention interventions remain at
chance. The reward-shuffled negative control remains at chance. This promotes
stable source-trust reuse from scalar outcomes for that audited runtime, not
arbitrary trust reversal, learned delay compensation, or natural multimodal
grounding.

The current v17 follow-up sequential reversal verifier is recorded under
`session_records/contradiction_amodal_2026-08-03/`. Its default random-reversal
harness was corrected so the documented four-block sequence can actually
generate a bounded reversal. Neutral source-credit initialization and
count-invariant source attribution make fixed reversal pass in one seed, and a
lower learning rate produces one passing 1024-update seed, but the three-seed
replication fails. Arbitrary temporal trust reversal therefore remains an
explicit implementation frontier rather than an implied consequence of the
fixed-role calibration result.

The original prototype path is:

```text
RGB frame -> VisionEventEncoder inside UnifiedCognitiveController
          -> recurrent controller/memory -> latent intention
          -> actuator inside the same model -> two action logits
```

The first behavior-preserving extraction rung now also provides:

```text
external VisionEventEncoder -> AmodalEvent -> controller.step_events()
    -> IntentEvent -> external ActionIntentDecoder
```

The three components own disjoint parameter sets and serialize independently.
The extracted path reproduces the real five-capability checkpoint exactly on a
64-lifetime audit: all 66 source tensors reconstruct exactly and maximum logit
difference is zero. The legacy API remains as a compatibility wrapper.

A promoted successor also removes the active compatibility suffix. It folds
the learned two-action residual through the frozen decoder's minimum-norm right
inverse into a 24-dimensional intention residual. This requires zero examples,
semantic labels, verifier outcomes, or optimizer updates. Across 12,288 paired
decisions it caused no action flips (maximum logit drift `5.72e-6`), and all
five repertoire gates passed on 4,096 held-out lifetimes. The external decoder
therefore receives a structurally zero compatibility suffix on the promoted
checkpoint.

The clean base intention now also fans out through a runtime-variable output
bus. A second opaque protocol backend was calibrated independently from only
its own attempted commands and scalar outcomes. Three seeds crossed the static
gate after 64 verifier bits, all passed the five-capability closed loop at 512
lifetimes, and the promoted decoder passed at 4,096. Simultaneous primary and
protocol outputs passed intention-shuffle, zero-intention, reward-shuffle, and
parameter-immutability controls.

On the input side, `AmodalEventCollection` now carries a padded event axis plus
a per-example presence mask and generic confidence. `AmodalInputBus` performs
permutation-invariant confidence attention and an optional learned set
residual. A structural diversity gate keeps that residual exactly zero for N=1
and identical duplicates, even after learning.

The first synchronous complementary N=2 task passes. Two separately encoded
partial views are individually at chance, but a 4,817-parameter generic bus
trained only from attempted actions and scalar outcomes reaches 96.46% when
their evidence is combined. Three seeds replicate bars acquisition after
768–1,344 verifier bits. One promoted seed also transfers above 90% to unseen
diamond and dot-pair renderings; that cross-renderer transfer is not yet
replicated at the strict gate.

The confidence-routing N=3 gate also passes without changing controller or bus
weights. On 4,096 held-out lifetimes, two streams score 96.58%; an opaque
third stream at confidence 1.0 scores 58.42%; the same stream at generic
confidence 0.01 scores 96.40%. This qualifies confidence-aware variable
cardinality routing, but the confidence value was supplied by the frontend.
A tiny head trained only on clean/corrupted latent consistency, replicated
across two seeds, improves 80%-corrupted N=3 behavior by 5.24–6.61 points
while preserving N=2 at 98.77–99.04%. This handles corruption/missing evidence,
not arbitrary valid distractor relevance.

A self-supervised same-frame pair-agreement head closes that relevance gap for
the audited N=3 setting. Across two seeds, a valid irrelevant third stream
improves from 57.45% to 89.25–89.59%, while N=2 remains 98.51–98.55%; useful
and distractor confidence separates by about 0.44. The positive relation is
same-frame complementary-view agreement, so cross-modality relevance and
larger N remain open.

The cardinality extension now uses a promoted hidden-64 agreement head trained
for 256 self-supervised updates. With two useful streams and up to nine valid
distractors, N=3 reaches 96.15–96.40%, N=4 95.76–96.13%, N=5 94.96–95.52%,
N=6 94.04–94.42%, N=7 92.77–93.17%, N=8 91.23–91.73%, N=9 89.43–89.96%,
N=10 87.45–88.10%, and N=11 85.34–86.14%, versus 49.6–58.3% without
agreement; N=2 remains 96.19–96.47%. N=12 falls to 83.19–83.88%, below
the pre-registered 85% gate, so N=11 is the current audited frontier. The
positive relation is still same-frame visual-view agreement; cross-modal
agreement and more than eleven streams remain open.

Timestamp-preserving transport alignment is now also qualified. At 4,096
lifetimes, out-of-order delivery and 0.25-unit timestamp jitter reproduced
synchronous actions exactly at 96.36%; genuinely mismatched timestamps stayed
in separate windows. This is transport plumbing, not learned delay handling.
Two outcome-only attempts to fine-tune the input bus against erased pixels
preserved clean accuracy but worsened held-out corruption accuracy, so the
frozen bus remains promoted. Robust noisy-stream handling now requires a
corruption-aware frontend or uncertainty mechanism with its own causal audit.

A protected three-stream token experiment now qualifies a stronger gradual
learning rung. Starting from a two-stream parent, a generic confidence gate
kept the new token stream out of the inherited bus until reward training, and
the frozen controller acquired the third temporal skill at 94.1% while vision
remained 100.0% and audio 62.4%. Reset, time-shuffle, and cross-episode token
swaps collapsed the new skill toward chance; all inherited controller,
vision, and audio tensors were unchanged. This is a synthetic token frontend,
not a natural-language qualification, and one independent seed failed, so
population/exploration stability remains open.

This is still **not yet the target architecture** because:

- synchronous event sets have been behaviorally qualified at N=1, N=2, and a
  confidence-gated N=3 distractor control;
- learned delay compensation, broad natural cross-modality relevance, fully
  learned missing-stream policies beyond the bounded verifier timeout, and
  arbitrary temporal trust reversal are not yet qualified; synthetic source
  reliability, context-dependent candidate relevance, context-conditioned
  contradiction arbitration, and stable source-trust reuse are promoted only
  within their audited verifier families;
- archived migration-v1 checkpoints retain two reserved compatibility
  coordinates, but the current v21 controller and intention schema do not
  expose an active action-residual path;
- the token-stream result is not yet an unseen natural-language or speech
  frontend/backend qualification; those adapters still require causal audits.

The synthetic token boundary is now qualified separately from the earlier
reward-trained token experiment. A fixed 16x16 RGB patch-token sensor and a
small frontend learned the frozen vision neural-IR basis from paired
encoded-event consistency only. Two independent 32-update runs passed, and a
saved-frontend 512-lifetime replay reached 98.32%/91.88%/94.69% fused on
bars/diamonds/dot-pairs, with shuffled partners near chance and contradictory
prediction flips of 93.59%/75.70%/90.59%. The controller and N=1 behavior were
unchanged. This qualifies another raw-stream adapter into the shared IR; it
does not qualify natural language, arbitrary tokenization, cold-start
alignment, or reward-only encoder learning. Reports and artifacts are in
`session_records/amodal_latent_alignment_2026-08-03/`.

The discrete text boundary is now qualified as a separate adapter. A fixed
sensor serialized each pooled RGB view into 768 meaningless symbols, and an
embedding frontend aligned those symbols into the same frozen neural IR using
paired consistency. A saved-frontend 512-lifetime replay reached
96.33%/94.41%/96.91% fused on bars/diamonds/dot-pairs, with shuffled partners
near chance and contradictory flips of 87.42%/78.48%/95.08%. This is text
transport plumbing, not natural-language semantics; the next language claim
requires a real text stream and a separate grounding audit.

A grounded-caption adapter now passes that narrower bridge. An external source
describes visible shape, colour, and location in a short word sequence, never
the hidden rule or correct action. Paired consistency aligns a small word
frontend into the frozen IR; a 512-lifetime saved replay reaches
98.36%/93.20%/95.16% fused on bars/diamonds/dot-pairs, with shuffled captions
near chance and contradictory flips above 78%. This is synthetic grounded
language transport, not a pretrained LLM or free-form natural-language claim.

The three qualified frontends now compose simultaneously through the same
input bus. Two independent 512-lifetime frozen-core audits of vision plus
synthetic audio plus discrete text reached 96.05--96.37% bars, 98.12--98.16%
diamonds, and 96.29--96.60% dot-pairs; shuffled triples stayed near chance,
contradictory flips stayed above 89%, and stream-order agreement was exactly
100%. This qualifies simultaneous N=3 transport and set permutation, but not
natural-language grounding, learned cross-modal relevance, or arbitrary N.

The first complete runtime composition API is now implemented as
`AmodalControllerRuntime`. It keeps a single extracted controller core, accepts
any nonempty mapping of registered raw frontends or already encoded events,
combines them through the opaque event bus, and fans the resulting intention
through a runtime-variable decoder mapping. Encoder and decoder registration
does not change controller parameter shapes. Simultaneous stream permutation,
pre-encoded sensor injection, missing-frontend rejection, and two-backend
fan-out are covered by tests. This is an interface/plumbing milestone, not a
claim that arbitrary natural modalities already learn or transfer.

The wrapper has now passed a behavioral audit using the promoted complementary
N=2 skill. With two independently registered external vision frontends and one
frozen controller, the 4,096-lifetime MPS audit reached 96.57% on bars, 91.13%
on diamonds, and 95.56% on dot-pairs; an independent 512-lifetime replica
reached 97.03%, 90.27%, and 95.66%. Individual streams and shuffled partners
remained near chance, contradictory partners caused the expected prediction
flips, and wrapper outputs were exactly equal to the prior explicit bus path
(maximum action-logit difference 0.0). This qualifies the runtime boundary for
an existing causal skill, not natural audio/language transfer or cold-start
cross-modal learning.

Therefore current results establish extracted vision-grounded neural IR,
cognition, memory, audited M-output fan-out, and synchronous complementary N=2
composition plus an explicit N-to-M runtime wrapper—not yet unrestricted
asynchronous or naturally multimodal operation.

A promoted outcome-only delayed rung now adds a narrower temporal result. With
stream `a` arriving at tick 0, stream `b` at tick 1, an opaque action at tick
2, and scalar reward/propensity feedback at tick 3, an independent 2,048-step
run reached 100% fused reward. Missing evidence and shuffled partners remained
near the expected 50% partial-information ceiling; contradictory evidence
collapsed to zero reward. This qualifies fixed-schedule asynchronous evidence
accumulation and delayed feedback through the clean runtime, not learned
wait/proceed decisions, recovery of absent evidence, natural sensor timing, or
persistent-memory utility. The optional memory arm filled its bounded store,
but clearing memory changed reward by only 0.15 percentage points. Evidence is in
`session_records/async_memory_amodal_2026-08-03/`.

The next memory rung promotes only scalar outcome recall. Across seeds 17, 18,
and 19, intact and persistent-replacement recall both reached 1.0, while clear
memory scored 0.4727/0.4922/0.4941 and value corruption scored
0.4922/0.5059/0.5371. An independent reward-randomized control remained at
0.5078. The write-strength path is trainable through the differentiable
transaction; ordinary commit rates were 1.0, 0.6563, and 1.0. This is a
causal memory-use result through the canonical controller, not general
episodic memory: learned skip or utility-based retention, multi-row content
addressing, batch isolation, and cross-adapter retrieval remain open. Evidence is in
`session_records/memory_recall_amodal_2026-08-04/`.

A fixed-write v17 two-slot follow-up under
`session_records/memory_binding_amodal_2026-08-04/` now qualifies the backend
scope contract only: identical keys in independent batch scopes remain
separate, including through differentiable transactions and snapshots. The
controller population was rejected for learned content binding: at 128 updates
the three seeds reached 0.5234/0.7266/0.5625 intact recall, while swapped-slot
recall reached 0.5313/0.8359/0.5625. Multi-row learned binding under
interference therefore remains an implementation frontier, not an implied
consequence of adding isolated memory banks.

The v18 follow-up under
`session_records/memory_binding_amodal_v18_2026-08-04/` promotes a narrower
controller result. A shared event-window address, independent of recurrent
state and feedback, lets two fixed-write rows remain separately retrievable
after state reset across four opaque batch scopes. Seeds 17, 18, and 19 all
reach 1.0 intact recall; clear, corruption, swapped-slot, and swapped-scope
controls stay near chance, and reward randomization remains at 0.5234. This
qualifies fixed-write two-slot binding and batch isolation, not learned skip or
utility-based retention, persistent episodic utility, or cross-adapter retrieval.

The first v20 cue-guided retention rung under
`session_records/memory_retention_amodal_v20_2026-08-04/` was rejected. After a
short single-slot curriculum, intact recall was `0.4922`, clear-memory
`0.5146`, corruption `0.4971`, and reversed-order `0.4902`; the
reward-shuffled control was near chance at `0.4863`. A bridged run with 1,024
single-slot updates reached a stable `1.0` parent prefix before adding the
distractor, but still ended at `0.5020` intact, `0.5049` clear, `0.5195`
corrupt, and `0.4805` reversed-order recall despite a `94.69%` write commit
rate. The pair-context write-policy path is therefore implemented and
checkpoint-versioned, but it has not demonstrated learned utility-based
retention. The next high-ROI work is an explicit phase-transition training
protocol and parent-capability protection, not a larger three-minute budget.

The v21 follow-up adds a generic latest-token-to-prior-token match feature and
an opt-in Bernoulli straight-through write sampler whose opaque log-probability
can receive outcome-only policy-gradient credit. The bridged target-first run
under `session_records/memory_retention_amodal_v21_2026-08-04/` again reached a
stable `1.0` parent prefix, but retention remained at `0.5010` intact versus
`0.4951` after clearing memory, with an `85.50%` durable write rate. The
`+0.0059` causal gap is far below the `+0.15` gate, so v21 is rejected as a
learned retention improvement. The sampler is retained only as training
infrastructure; no v21 retention weights or capability claim are promoted.

The corrected follow-up keeps the query cue visible at recall and promotes no
new capability on that basis. Runtime v23 used a transport-augmented
latest-event address,
the generic latest-prior cosine context, parent-stability gating, balanced
target-first/target-last warmup, held-out validation selection, and missing-cue
controls. The three-seed balanced population in
`session_records/memory_retention_amodal_v32_2026-08-04/` is rejected: seed 17
remains at chance, seed 18 is strong only when the target arrives first, and
seed 19 is the only seed that passes both order checks. The resulting
order-dependent behavior is a last-write/first-write shortcut, not stable
cue-conditioned utility. A first-to-earliest-token variant was tested and
removed after collapsing to chance. No v23 retention weights are promoted; the
next bottleneck is lower-variance outcome-only credit assignment, not another
permanent address feature. A discarded private write-policy probe confirms the
mechanism: all three seeds saturate target and distractor write strengths near
one and commit both, so the immediate next test is utility-policy stabilization
with generic write regularization.

The parent-stable v33 mini-rung tested the existing generic write-cost term.
The zero-cost single seed reached `0.884` target-first and `0.869` target-last,
but the `0.02` cost arm fell to `0.749` and `0.753` and failed cue gain. This
does not justify a population claim or a new permanent branch; write cost alone
is rejected as the fix. Reports and the accounting ledger are in
`session_records/memory_retention_amodal_v33_2026-08-04/`.

A v34 candidate that added generic memory-read similarity/hit features to the
write head preserved the parent audit but collapsed retention to chance. It is
removed; the v24 schema is reserved for the later stable-address correction,
not this rejected branch. The remaining high-ROI problem is credit assignment,
not additional memory metadata.

The v35 batch-size-64 control also remained at chance, and the v36
parent-protection control still produced a target-last shortcut (`0.764`) over
target-first (`0.506`). These controls rule out simple trajectory variance and
destructive parent co-adaptation. The remaining implementation bottleneck is
specifically lower-variance outcome-only credit for conditional writes.

The v37 write-critic baseline and v38 hard-retention control both reproduce
the same last-write shortcut (`0.522` target-first, `0.997` target-last). They
are retained only as opt-in training infrastructure. The next mechanism must
provide counterfactual credit for the write action itself; another scalar
baseline or memory metadata field is not justified.

The v39 counterfactual write-utility protocol resolves that specific training
bottleneck at the sub-minute qualification rung. For one randomly selected
event position, paired common-random arms force write versus skip while all
other positions use shared sampled writes. The generic write logit receives
only the scalar recall difference between the arms; verifier bits, target
indices, and intervention metadata remain trainer-private. With a neutral
write-policy reset and no parent freeze, seeds 17, 18, and 19 pass the
retention gate: mean intact recall is `0.956`, clear-memory `0.511`,
corrupt-memory `0.488`, reversed-order `0.954`, target-first `0.954`, and
target-last `0.940`. Random-action recall is `0.505`, and the reward-shuffled
control remains at chance. This qualifies narrow learned cue-conditioned
utility-based retention and promotes the training protocol, not a checkpoint.
The next evidence rung is a roughly three-minute replication with retention
on mastered primitives and fresh-learner transfer; persistent-memory utility
and cross-adapter retrieval remain unqualified. Full accounting is in
`session_records/memory_retention_amodal_v39_2026-08-04/`.

The longer v40 stress test exposed a second bottleneck: strong retention could
forget the mastered single-event primitive (`0.738`). It is rejected. v41
adds one ordinary outcome-only parent rehearsal update per retention update and
selects only validation states that preserve both the retention gate and parent
retention. The unprotected three-seed population passes stable-prefix
accounting with mean intact `0.995`, clear `0.512`, corrupt `0.493`, reversed
`0.994`, target-first `0.993`, target-last `0.998`, and mastered-primitive
retention `0.991`; the reward-shuffled control remains at chance. This
promotes the parent-preserving training protocol and narrow sub-minute
retention behavior, not a checkpoint. The next implementation rung is longer
rehearsal-preserving replication plus a matched fresh-learner transfer curve;
persistent-memory utility remains unqualified. Evidence is in
`session_records/memory_retention_amodal_v40_2026-08-04/` and
`session_records/memory_retention_amodal_v41_2026-08-04/`.

The v42 2,048-update stress test preserved the mastered primitive but failed
the stable validation-prefix rule after continued training. It is rejected.
v43 adds an explicit consolidation stop after three consecutive held-out
validations pass, stopping at 320 retention updates for seed 19 while
preserving `1.000` parent retention and `1.000` unseen-token retention. A
matched fresh-learner transfer curve reaches stable threshold at 20,480 bits
versus 13,312 for the retained learner (`1.538x` fresh-over-transferred).
This is a one-seed transfer lead, not a population or checkpoint promotion.
The next high-ROI work is transfer-ratio replication across seeds, followed by
persistent-memory write/reload/corruption qualification. Evidence is in
`session_records/memory_retention_amodal_v42_2026-08-04/` and
`session_records/memory_retention_amodal_v43_2026-08-04/`.

The v44 one-seed boundary audit writes and reloads a learned retention episode
through `PersistentContentAddressedMemory`, preserving `1.000` intact recall.
It also reproduces `1.000` mastered-primitive retention, `0.996` zero-shot
unseen-token retention, and a `1.538x` fresh-over-transferred stable-bit ratio.
This qualifies the persistent-memory interface for the narrow verifier only;
multi-seed reload, corruption recovery, transfer retention, and checkpoint
promotion remain outstanding. Evidence is in
`session_records/memory_retention_amodal_v44_2026-08-04/`.

The v46 population audit closes the storage-boundary gap for the narrow
verifier across seeds 17, 18, and 19: reload averages `0.991`, every
checksum-invalid snapshot is rejected, and atomic restoration returns `1.000`
recall for every seed. This qualifies persistence as a replaceable runtime
boundary, not as a general episodic-memory capability. Retention itself
promotes two of three seeds; seed 18 fails the stable-prefix rule. Transfer
also remains unresolved: seed 19 reproduces `1.538x` fresh-over-transferred
stable bits, while seeds 17 and 18 do not reach a stable threshold under the
matched short budget. No checkpoint is promoted. Evidence is in
`session_records/memory_retention_amodal_v46_2026-08-04/`.

The v47–v48 transfer-control diagnostics show why the missing population
ratio cannot be treated as a bookkeeping detail. For promoted seed 17, the
transferred learner reaches stable threshold at `28,672` bits, while a matched
fresh learner fails parent qualification even after 2,048 phase-1 updates and
therefore never enters retention training. Fresh-parent acquisition must be a
first-class transfer gate, with multiple fresh initialization seeds, before a
finite transfer ratio or reusable-capability claim is made. Evidence is in
`session_records/memory_retention_amodal_v47_2026-08-04/` and
`session_records/memory_retention_amodal_v48_2026-08-04/`.

The v49 three-initialization transfer control confirms the gate is necessary:
only one of three fresh seed-19 learners qualifies its parent. That learner
reaches `20,480` stable bits against `13,312` for the transferred learner,
but the other two never enter retention training, so the population transfer
status is `fresh_parent_not_qualified` and no ratio is promoted. The transfer
boundary must be stabilized across fresh initializations before it can support
a reusable-capability claim. Evidence is in
`session_records/memory_retention_amodal_v49_2026-08-04/`.

The v50 value-baseline diagnostic is not promoted. A training-only critic
reduced some fresh-parent failures but destabilized the retained seed-19
retention gate, reducing mastered-primitive retention to `0.500`; the
transferred parent also failed qualification. The critic is now opt-in and
excluded from the default protocol/checkpoint. The next implementation target
is lower-variance parent acquisition whose phase transition does not regress
retention. Evidence is in
`session_records/memory_retention_amodal_v50_2026-08-04/`.

The v51–v52 parent-action diagnostics are also rejected. Coupled forced
actions fail parent qualification; adding a fixed-write scaffold improves
fresh acquisition to two of three initializations but regresses mastered
primitive retention to 0.773 and produces no stable threshold. This narrows
the unresolved implementation bottleneck to phase-transition co-adaptation
between parent action behavior and the learned write policy. The original
policy-gradient parent protocol remains the default control. Evidence is in
`session_records/memory_retention_amodal_v51_2026-08-04/` and
`session_records/memory_retention_amodal_v52_2026-08-04/`.

The v53 mixed protocol uses fixed-write action credit only for initial parent
acquisition and ordinary rehearsal afterward. It preserves mastered-primitive
retention and reaches a stable threshold, but corrupt-memory recall rises to
0.519, failing the causal gap gate; transfer remains unqualified. The
phase-transition co-adaptation problem therefore remains open, and the
original policy-gradient protocol stays the default. Evidence is in
`session_records/memory_retention_amodal_v53_2026-08-04/`.

The v54–v55 phase-transition controls close two implementation hypotheses
without promoting either. v54 now clears stale Adam moments when a neutral
write-policy reset occurs, but the matched reset arm reaches threshold later
(`23,040` versus `17,920` verifier bits). v55 freezes the generic write policy
during parent acquisition and unfreezes it at retention; it matches rather
than improves the unfrozen control. The v56–v57 recall-only parent-credit
controls remove an unidentifiable probe-action policy-gradient term. Applying
that intervention during rehearsal damages parent retention; applying it only
during acquisition produces transient gains but fails the stable-prefix rule.
These diagnostics are rejected, the default policy-gradient protocol is
unchanged, and transfer remains the active bottleneck. Evidence is in
`session_records/memory_retention_amodal_v54_2026-08-04/` through
`session_records/memory_retention_amodal_v57_2026-08-04/`.

The v58 feedback-residual diagnostic improves per-run parent acquisition and
preserves narrow retention, but the three-seed transfer population is still
unqualified. The v59 identity-address diagnostics are seed-dependent and are
rejected. Their mechanistic value is that they expose a boundary violation in
the prior address: the same event acquired different keys when its transport
age changed between write and recall.

Runtime v25 corrects that violation. The canonical memory address is a shared
learned projection of the latest event payload only; age, duration, timestamp
presence, and confidence remain available to controller reasoning and the
generic write utility policy but cannot alter the durable key. v23 checkpoints
are accepted as legacy and retain their transport-augmented behavior after
migration. The v60 three-seed audit passes narrow retention and persistence,
but only two seeds qualify transfer, so this is an interface correctness fix,
not a learned population-transfer claim. v61 shows that fixed token identities
were a second shortcut: token-diverse retention gives all three retained
models `0.996–1.000` unseen-token recall, but the fresh transfer population
remains unstable. The v62–v64 budget and warmup controls do not close that
gap. The v65 no-feedback ablation fails causal retention and persistent reload
despite `1.000` unseen-token recall, so the generic opaque
feedback-to-memory-value residual is canonical in runtime v25. Evidence is in
`session_records/memory_retention_amodal_v58_2026-08-04/` through
`session_records/memory_retention_amodal_v65_2026-08-04/`.

The v66 transfer audit found a harness mismatch: fresh and transferred arms
silently disabled the declared retention write-policy reset and omitted other
training controls. The transfer runner now forwards the complete declared
configuration to both matched arms. v67–v71 then isolate phase-transition
variance. A bounded reuse of each randomized opaque token pair for four
episodes keeps the outcome-only policy target stationary without adding a
semantic or modality-specific branch.

The v72–v73 1,024/1,024 three-seed rung passes the narrow main retention gate,
four-pair unseen-token minimum, persistent reload/checksum recovery, and
matched fresh transfer for seeds 17, 18, and 19. Minimum unseen-pair recall is
`0.879`, `0.727`, and `0.840`; fresh-over-transferred stable-bit ratios are
`2.103x`, `1.538x`, and `2.000x`. This is a promoted outcome-only
retention/transfer capability for the narrow verifier. It is not evidence of
natural-language semantics, physical control, or general episodic memory.
The four-pair audit and full accounting are in
`session_records/memory_retention_amodal_v73_2026-08-04/`.

The original v74 three-slot/two-row follow-up is retained as a superseded
harness record. Its balanced counterfactual arms duplicated verifier rows
without duplicating the trainer's target-position assignment, so its
position-specific metrics are confounded. The corrected v76 rung fixes that
pairing and promotes the next boundary: the v27 controller combines a shared
learned memory projection with a residual learned-event identity path, and its
write utility receives strongest-prior event binding. Across seeds 17, 18, and
19, intact recall is `0.924/0.997/0.999`, target-first is
`0.998/0.997/0.998`, target-last is `0.846/0.999/0.997`, and unseen-token
minimum recall is `0.914/0.836/0.930`. Persistent reload is
`0.934/0.992/0.996`, recovery is `0.938/1.000/1.000`, and checksum corruption
is rejected for every seed. This remains an outcome-only bounded-memory claim,
not natural-modality grounding or general episodic memory. Evidence is in
`session_records/memory_retention_amodal_v76_2026-08-04/`.

The v75 synthetic cross-adapter rung qualifies the two-row neural-IR boundary.
The v77 three-seed rung extends it to three outcome-only rows with an opaque
target cue, cued-row-last presentation, and persistent reload/recovery. Fresh
aligned-reader minima are `0.996`, `0.997`, and `1.000`; aligned-vs-raw mean
gains are `0.496`, `0.505`, and `0.515`; swapped-slot maxima are
`0.511`, `0.509`, and `0.520`; and checksum corruption is rejected for every
seed. This qualifies
synthetic replaceable neural-IR cross-adapter retrieval, not natural-modality
grounding or general episodic utility. The three-slot/two-row bounded-
interference variant also passes after the memory backend separates strict
write collision matching (`0.95`) from learned-IR read matching (`0.75`) and
returns no value for near-miss rows. Evidence is in
`session_records/cross_adapter_memory_amodal_v77_2026-08-04/`.

The v78 follow-up removes the recency shortcut: an opaque cue arrives before
three rows, but the target row is presented at a randomized position. A
trainer-only counterfactual leave-one-out write intervention stabilizes the
generic policy across seeds 17, 18, and 19. Fresh-reader minima are
`0.991/0.988/0.998`; fresh aligned-vs-raw gain minima are
`0.476/0.458/0.482`; and persistent reload, recovery, checksum rejection,
clear-memory, corruption, and swapped-row controls pass. Cue removal and cue
swapping remain diagnostics, not a cue-conditioned selection claim; capacity
one and general episodic memory remain open. Evidence is in
`session_records/cross_adapter_memory_amodal_v78_2026-08-04/`.

The v79 capacity-one follow-up found a separate implementation bottleneck in
the experiment harness: action sampling had advanced the event window, then
the outcome-bearing write advanced it again for the same payload. The corrected
preview/commit lifecycle inserts each learned event once, so bounded windows
retain earlier cues. Stable-content prior binding keeps timing and payload
identity separate, while the one-slot intervention isolates downstream writes
when assigning generic retention utility. The corrected rung reaches strong
main and causal-control recall across three seeds, but two seeds remain below
the fresh-token population gate. This qualifies the lifecycle and causal
mechanism fix; capacity-one population promotion and general episodic memory
remain open. Evidence is in
`session_records/cross_adapter_memory_amodal_v79_2026-08-04/`.

The v80 rung closes the remaining capacity-one population gap by making the
trainer-only counterfactual write intervention token-diverse, with the same
four-episode opaque-token reuse schedule used during base acquisition. All
three seeds pass the fresh-token, causal-control, persistent-reload, and
reward-shuffled gates. This promotes synthetic outcome-only capacity-one
cross-adapter retrieval with randomized target position; natural-modality
grounding and general episodic memory remain open. Evidence is in
`session_records/cross_adapter_memory_amodal_v80_2026-08-04/`.

The working-memory pressure test now qualifies a narrow frozen-controller
growth boundary. A slot-free parent receives a zero-output generic successor
state; only that state is trained from rendered events, opaque attempted
actions, and scalar verifier outcomes. Parent-logit distillation on fresh
rehearsal streams protects inherited behavior, while
`ExecutableArtifactMemory` stores and reloads the tensor-only growth artifact
through `load_growth_artifact`. Two seeds improve an adjacent complement
primitive by `+32.03/+33.59` percentage points, zeroing the artifact returns
to the parent, the shared controller digest remains unchanged, rehydration
matches the live child, and spans 4/6/8 remain within the two-point retention
gate. The unprotected control loses 3.91 points on span eight, establishing
the retention failure that the distillation term repairs. A reward-shuffled
control still gains `+7.03` points, far below aligned training, so exclusive
reward attribution remains unqualified.

This is not yet cold-start address discovery, arbitrary procedure induction,
variable-capacity mastery, or general cognition. The result is recorded in
`session_records/sequence_working_memory_2026-08-02/frozen_growth_complement_distill_2026-08-04/`.

The initial raw-byte language follow-up is retracted. Its caption renderer
used verifier-generated context IDs to choose a shape word, so the text stream
was not pixel-only. The renderer has been repaired to derive descriptors from
pixels. The corrected character n-gram CNN passes the three-seed training-time
gate, and all three independent saved frontends pass a 1,024-lifetime replay.
This promotes synthetic pixel-grounded UTF-8 event transport into the frozen
amodal boundary, not open-ended natural-language understanding. The evidence
is retained under
`session_records/natural_text_grounding_pixel_only_2026-08-04/`.

The follow-up external-corpus boundary is promoted separately. A versioned,
independently authored phrase corpus with two training variants per style
passes all three 1,024-lifetime saved-frontend replays using the same frozen
controller and byte frontend. This establishes controlled corpus-backed UTF-8
event transport, not open-world language understanding. Evidence is under
`session_records/natural_text_grounding_external_corpus_v2_2026-08-04/`.

The next caption boundary is also promoted. Three independently trained byte
frontends consume a versioned static annotation table whose entries are
complete pre-authored sentences, rather than runtime templates with slots.
The pixel-only source performs only the image-to-annotation join; no verifier
metadata or semantic target enters the learner. At 1,024 fresh lifetimes per
appearance/style cell, the three saved frontends reach minimum fused accuracy
of `92.83%`, `91.39%`, and `92.15%`, with shuffled partners at most `54.61%`,
contradictory partners at most `16.27%`, contradiction flips at least
`75.43%`, and full-vision retention at least `97.54%`. This promotes static
pre-authored caption transport into the frozen amodal neural IR. It remains a
controlled synthetic visible-scene annotation benchmark, not open-world
language understanding, speech, or general semantic reasoning. Evidence is in
`session_records/natural_text_grounding_external_annotation_table_v3_2026-08-04/`.

## Routed artifact compaction boundary (2026-08-05)

The next memory bottleneck was not append capacity but safe compaction of
independently learned executable artifacts. A naive merge executed both
growth slots at once and failed behavior preservation. The corrected contract
stores the merged tensor namespaces in one physical row, keeps multiple opaque
address aliases, and returns an opaque view identifier with the verified
memory handle. The generic caller then projects only the selected namespace
into the frozen growth boundary; the memory backend does not interpret the
view.

Two 512-update seeds passed the held-out promotion gates: source rows `2`
became one row, both aliases routed to their distinct views, parent retention
was `1.000`, the frozen-core digest was unchanged, persistent reload matched,
checksum corruption was rejected, and the rejected candidate was not adopted.
No consolidation optimizer updates or replayed examples were used. This
promotes routed logical compaction, not byte compression, arbitrary new
computation, unrestricted procedure induction, or general continual learning.
Independent capabilities remain append-only unless a behavior verifier admits
a compact routed representation. Evidence is in
`session_records/sequence_working_memory_2026-08-02/artifact_consolidation_v1_2026-08-05/`.

## Outcome-only view routing boundary (2026-08-05)

The caller-supplied view shortcut is now removed from the narrow pressure test.
`view_candidates()` exposes only opaque alias keys and view tokens, while a
replaceable `FactorizedOpaqueAddressRouter` learns the view choice from
controller-produced queries and paired attempted-view scalar outcomes. The
router receives no task or span identity and is permutation-equivariant over
candidate views.

Across two seeds, held-out route accuracy and candidate-permutation accuracy
were both `1.000`; reward-shuffled routing was `0.438/0.500`; wrong-view
behavior was causally worse for both procedures; reload, exact-candidate,
checksum, and frozen-core gates passed. This promotes learned routing between
already-acquired views, not arbitrary task discovery, unbounded executable
program induction, or general continual learning. Evidence is in
`session_records/sequence_working_memory_2026-08-02/artifact_view_routing_v1_2026-08-05/`.

## Four-view routing scaling boundary (2026-08-05)

The two-view routing result scales to four independently acquired span-4
procedures in one physical artifact row. Four opaque storage identities are
learned through a joint permutation-equivariant scorer with paired
counterfactual scalar credit; the router query is the controller’s learned
memory-query representation after the first query event. No operation name,
task ID, or correct unattempted choice reaches the router.

Across two seeds, route/permutation accuracy was `1.000/1.000` and
`0.969/0.969`; reward-shuffled routing was `0.215/0.250`; all four procedures
passed mastery and wrong-view causal gates; reload, checksum, exact-candidate,
and frozen-core controls passed. This promotes bounded four-view scaling only.
Context-derived address collision, factorized routing, and direct attempted
outcome credit remain rejected controls. Evidence is in
`session_records/sequence_working_memory_2026-08-02/artifact_view_routing_scaling_v1_2026-08-05/`.

## Outcome-gated online view growth boundary (2026-08-05)

The first five-view online-growth audit exposes the next important boundary.
Four learned executable views are acquired and routed, then their router and
the controller core are frozen. A fifth `rotate` view is added to the same
physical artifact row. A zero-initialized memory-side
`OpaqueViewRouteExtension` learns only from fresh paired scalar outcomes for
the new procedure; it does not replay old route examples or update the old
router.

Optimistic extension preemption is rejected: a frozen closed-set router can
assign high confidence to the wrong old view for a genuinely new procedure.
The promoted selector uses an outcome-gated fallback instead. Existing routes
retain priority; after an opaque old attempt fails, the new external view is
opened and can recover the procedure. Across seeds 69316 and 69317, old-route
retention was `1.000/0.988`, failure-gated new-view recovery was `1.000/1.000`,
combined five-view routing was `1.000/0.994`, permutation accuracy matched,
old false-positive rate was `0.000/0.000`, and the reward-shuffled extension
selected the new view `0.000/0.000`. Reload, exact-candidate, corruption,
frozen-core, frozen-router, and wrong-view causal gates passed with zero
replayed examples after extension.

This promotes safe external capability addition with a bounded one-failure
cold start. It does not establish immediate novel-task routing, unrestricted
continual learning, arbitrary new computation, or unbounded memory growth.
Evidence is in
`session_records/sequence_working_memory_2026-08-02/online_view_growth_v1_2026-08-05/`.

## Two-step replay-free view-growth boundary (2026-08-05)

The next audit composes two sequential additions on the same frozen base. A
`rotate` view is added after the four known routes, then a
`complement_rotate` view is added after the first extension is frozen. The
second procedure must first fail through the old router and then through the
first extension before the second external view is opened. Both new views are
stored as opaque aliases in one physical artifact row.

Across seeds 69316 and 69317, old-route retention was `1.000/0.984`, both
new-view route accuracies were `1.000/1.000`, and the complete two-step chain
was `1.000/0.995`. Candidate permutation accuracy matched. The first
extension was selected on the second procedure at `1.000/1.000`; reward-
shuffled first- and second-extension selection was `0.000/0.000`. Behavior,
wrong-view causal, exact-candidate reload, checksum, frozen-core, frozen-first
extension, and zero-replay gates all passed.

This promotes a bounded two-step outcome-gated fallback and replay-free
external consolidation. It is evidence that isolated memory-side growth can
survive a second addition without modifying the controller or replaying prior
route examples. It is not general continual learning: the number of additions
is bounded, the new procedures are supplied by an external artifact trainer,
and arbitrary new computation, open-ended discovery, learned compression, and
unrestricted memory growth remain unqualified. Evidence is in
`session_records/sequence_working_memory_2026-08-02/multistep_view_growth_v1_2026-08-05/`.

## Three-step replay-free view-growth boundary (2026-08-05)

The cumulative external fallback chain now survives a third sequential
addition. After the frozen four-view base, `rotate`, `complement_rotate`, and
`adjacent_xor` are acquired as views `4`, `5`, and `6`. Each later procedure
first passes through the old router and every earlier extension as a failed
opaque attempt. All seven views remain isolated aliases in one physical
artifact row.

Across seeds 69316 and 69317, old-route retention was `1.000/0.992`, each of
the three new-view routes was `1.000/1.000`, and the complete three-step chain
was `1.000/0.998`. Candidate permutation accuracy matched. Every prior-
extension attempt rate was `1.000` on later procedures and reward-shuffled
selection of every new view was `0.000` on both seeds. Behavioral, causal
wrong-view, exact reload, checksum, frozen-core, frozen-extension, and
zero-replay gates all passed.

This promotes a bounded three-step outcome-gated fallback and replay-free
external consolidation. It demonstrates that isolated memory-side growth can
survive a third addition without controller updates or replay of earlier route
examples. It remains bounded: the external trainer supplies the new artifacts,
and unrestricted memory growth, learned compression, arbitrary new
computation, open-ended discovery, and general continual learning remain
unqualified. Evidence is in
`session_records/sequence_working_memory_2026-08-02/three_step_view_growth_v1_2026-08-05/`.

## Behavior-verified fixed-capacity artifact compression (2026-08-05)

The seven-view chain now has a real payload-capacity result rather than only
logical one-row compaction. A caller-owned float16 growth-artifact codec casts
the complete tensor payload before transactional promotion, and the frozen
growth loader explicitly opts into casting it back. The memory backend remains
opaque to the codec and performs the same integrity checks.

Across seeds 69316 and 69317, raw tensor payload bytes fell from `202,944` to
`101,472` (`0.500`) and serialized artifact bytes fell from `212,863` to
`111,167` (`0.522`). Compressed and uncompressed behavior were identical for
all seven selected views in both audits. Compressed wrong-view causal,
exact-alias, reload, checksum, frozen-core, frozen-extension, and zero-replay
gates passed.

This promotes behavior-verified fixed-capacity tensor compression for the
bounded seven-view chain. It is a storage codec and does not add computation;
learned compression, arbitrary new computation, open-ended memory growth, and
general continual learning remain unqualified. Evidence is in
`session_records/sequence_working_memory_2026-08-02/three_step_view_compression_v1_2026-08-05/`.

## Behavior-verified int8 artifact quantization (2026-08-05)

The seven-view payload now survives a stronger per-tensor symmetric int8
codec. Each quantized tensor carries an explicit positive scale; decompression
is caller-owned and occurs before the strict frozen-growth loader. The memory
backend stores only opaque tensor mappings and integrity hashes.

Across seeds 69316 and 69317, raw payload bytes fell from `202,944` to
`50,848` (`0.2506`), and serialized bytes fell from `212,863` to `69,771`
(`0.3278`). Quantized behavior stayed within the predeclared five-point
retention tolerance and above the behavior floor for all seven views on both
seeds. Wrong-view causal separation, exact aliases, reload, checksum,
frozen-core, frozen-extension, and zero-replay gates all passed.

This promotes behavior-verified int8 storage quantization for a bounded
external artifact chain. It is a replaceable storage codec, not learned new
computation; learned compression, arbitrary new computation, open-ended
memory growth, and general continual learning remain unqualified. Evidence is
in `session_records/sequence_working_memory_2026-08-02/three_step_view_quantization_v1_2026-08-05/`.

## Behavior-verified packed int4 artifact quantization (2026-08-05)

The same seven-view artifact now survives a packed signed-int4 codec. Each
floating tensor is quantized per output row, two int4 values are packed per
byte, and explicit scale and shape entries make the representation reversible.
Decompression remains caller-owned and occurs before the strict frozen-growth
loader; no controller or memory interface changes are required.

Across seeds 69316 and 69317, raw payload bytes fell from `202,944` to
`30,184` (`0.1487`), and serialized bytes fell from `212,863` to `58,007`
(`0.2725`). The complete three-step route chain remained `1.000/0.998`, with
minimum packed behavior `0.7227/0.7305`; packed behavior, wrong-view causal
separation, exact aliases, reload, corruption rejection, frozen-core,
frozen-extension, and zero-replay gates all passed.

This promotes behavior-verified packed int4 storage quantization for a bounded
external artifact chain. It is a replaceable storage codec, not learned
compression, arbitrary new computation, open-ended memory growth, or general
continual learning. Evidence is in
`session_records/sequence_working_memory_2026-08-02/three_step_view_int4_v1_2026-08-05/`.

## Replicated episodic context and causal credit (2026-08-05)

The next continual-learning boundary now has a reusable memory-side context
and credit contract. `EpisodicContextEncoder` consumes ordered learned event
tokens, opaque actions, scalar outcomes, and presence. Augmented episode views
train context without task labels; paired common-random write utilities train
per-event credit. An opaque route then learns candidate addressing from
attempted-row outcomes, while a fresh extension is added with the old router
and context encoder frozen.

In a three-procedure temporal audit whose single-event statistics are identical,
the recurrent context routed the two old procedures at `0.9688/1.000` versus
`0.500/0.500` for pooled events. Candidate permutation, new-route recovery,
new-route ablation, decisive-position credit, old-route retention, shuffled
outcome rejection, and zero-replay gates all passed across two seeds.

This promotes a bounded episodic-context and counterfactual-credit mechanism.
It does not establish unrestricted memory growth, arbitrary program
induction, natural-modality learning, or general continual learning. Evidence
is in
`session_records/sequence_working_memory_2026-08-02/episodic_context_credit_v1_2026-08-05/`.

The canonical Brain Workshop pressure test then reproduced a stronger narrow
reader mechanism. `OnlineEpisodicRelationReader` keeps a bounded external event
window and learns content-and-age retrieval over learned events, opaque prior
actions, and scalar prior outcomes. With the controller, event frontend, and
keypress feedback encoder frozen, a reward-only 32-update n-back-2 acquisition
reached `0.9375/1.0000/1.0000` fresh accuracy across seeds 17, 18, and 19;
time-shuffled controls were `0.4818/0.4792/0.5365`, history-reset controls
were `0.5000/0.5000/0.5000`, and zero replay was used. An explicit
post-acquisition audit protected all three opaque capability addresses. This
promotes bounded causal relation reading under a frozen core, but not n-back-3
transfer, sequential isolated growth, learned eviction, unrestricted memory
growth, or general continual learning. Evidence is in
`session_records/brainworkshop_canonical_2026-08-05/relation_reader_nback2_seed*_32_retention.json`.
The longer 512-update continuation replicated at `1.0000` fresh accuracy for
all three seeds, retained `1.0000` across three repeated fresh audits per seed,
and kept time-shuffle/history-reset controls at chance with zero replay. This
closes the bounded single-capability stability rung. The unresolved next
boundary is adding a second capability while the first external reader,
adapter, and retention record remain frozen and behaviorally protected.

The canonical runner then replicated that append boundary across seeds 17, 18,
and 19. After 128 n-back-2 updates, a second relation-reader slot was appended
and trained for 256 n-back-3 updates. Old-slot fresh accuracy remained
`1.0000/1.0000/1.0000`; new-slot fresh accuracy reached
`0.8042/0.8042/0.8021`. Time-shuffle controls stayed below `0.544` and
history-reset controls were `0.5000`; both opaque retention records were
protected, the old controller/reader/adapter/decoder hash was unchanged, and
all updates used zero replay. A 25% post-feedback slot exploration policy was
propensity-accounted to supply acquisition data without task labels. This
promotes bounded two-slot sequential growth under a frozen processor, not
unrestricted growth, learned eviction, reversal recovery, persistence, or
general continual learning. Evidence is in
`session_records/brainworkshop_canonical_2026-08-05/sequential_nback2_to_nback3_seed*.json`.

The first three-slot n-back-2 to n-back-3 to n-back-4 ladder then isolated the
next blocker. Forced candidate audits reached `1.0000` for all three slots and
all retention records remained protected, but dynamic fresh routing was
`1.0000/0.8042/0.5703`; the n-back-4 slot was selected for only `16.97%` of
training positions. The third capability therefore learned its candidate
behavior, while the opaque route failed to discover it reliably. This rung is
rejected pending an outcome-trained route learner that preserves old slots.
Evidence is in
`session_records/brainworkshop_canonical_2026-08-05/capacity_ladder_nback2_to_nback4_seed17.json`.
An outcome-trained proactive route head improved n-back-4 discovery to
`0.8828` but interfered with old n-back-2/3 routes, reducing them to
`0.6771/0.5167`; it is rejected. New-family scalar rewards alone cannot safely
reshape shared route selection. The next route rung needs an explicit
old-route-preservation invariant or persistent task-inference state.
A 12-step lifetime control raised safe failure-gated n-back-4 routing to
`0.7513` while preserving forced candidate retention, showing that more scalar
evidence helps but does not close the short-lifetime route-inference gap.

## Persistent route-evidence gate and long-lifetime growth (2026-08-05)

The canonical Brain Workshop route now has an external
`PersistentOpaqueRouteEvidence` ledger with a candidate-specific stable-prefix
promotion gate. The ledger stores only opaque slot indices, scalar outcomes,
and versioned statistics; it does not receive task IDs, n-back values,
semantic labels, or correct unattempted actions. Controlled route exploration
remains propensity-accounted, and the controller plus all prior slots stay
frozen while a new slot is acquired.

The eight-step short-lifetime ladder remains rejected at fresh
n-back-2/3/4=`1.0000/0.8042/0.5703`. This is an identifiability boundary, not a
candidate-computation failure: the cue-free verifier gives every n-back family
the same observable symbol-stream distribution, so a route cannot know the
correct opaque expert before the first scored feedback. A persistent global
preference is therefore unsafe on task switches even when it improves the
latest task.

With 16-step lifetimes, the same frozen-core route passed across seeds 17, 18,
and 19: fresh n-back-2/3/4=`1.0000/0.9231/0.8333`, time-shuffle and
history-reset controls stayed near chance, every forced candidate audit was
`1.0000`, prior-slot hashes were unchanged, and replay was zero. This promotes
bounded three-slot external growth for long enough lifetimes only. It does not
promote arbitrary new-task routing, short-lifetime mastery, learned eviction,
unrestricted memory growth, or general continual learning. Evidence is in
`session_records/brainworkshop_canonical_2026-08-05/capacity_ladder_persistent_route_steps16_seed17.json`,
`...seed18.json`, and `...seed19.json`.

## Context-conditioned route memory (2026-08-05)

The short-lifetime route gap is now split into its information and memory
components. `PersistentOpaqueContextRouteEvidence` indexes opaque candidate
slot ledgers by learned event keys. It never receives a task ID or semantic
field. Route reads are available online, while persistent writes are limited to
an explicit candidate-calibration transaction after forced scalar audits;
exploratory acquisition outcomes cannot silently rewrite the route prior.

An optional rendered cue token is sent through the ordinary learned event
frontend. Across seeds 17, 18, and 19, the 8-step cue-conditioned ladder
reached fresh n-back-2/3/4 accuracy `1.0000/1.0000/1.0000`, with near-chance
time-shuffle and history-reset controls, perfect candidate audits, causal
cue-shuffle separation, unchanged prior slots, a frozen controller, and zero
replay. This promotes bounded cue-conditioned route selection and validates
the external context-memory contract.

The cue-absent control remains the prior cold-start result, approximately
`1.0000/0.8042/0.57--0.58`; the same symbol-stream distribution does not
identify the correct expert before first feedback. Therefore this does not
establish hidden-task inference, arbitrary new computation, unrestricted
memory growth, or general continual learning. Evidence is in
`session_records/brainworkshop_canonical_2026-08-05/context_route_short_lifetime_seed17.json`,
`...seed18.json`, and `...seed19.json`.

## Nonstationary same-cue capability replacement (2026-08-05)

The external route boundary now survives a same-cue capability change. A
rendered cue is first calibrated to an n-back-2 slot; a new n-back-4 slot is
then acquired from fresh scalar outcomes with the controller and old slot
frozen. An explicit candidate-calibration transaction promotes the new slot
for the same learned cue, while a forced audit retains the old slot at
`1.0000`.

Across seeds 17, 18, and 19, new-route fresh accuracy was `1.0000` in every
case, cue-shuffled controls were `0.7578/0.7604/0.7708`, route-state reload
selected the new slot, prior state hashes were unchanged, and replay was zero.
This promotes bounded nonstationary external route replacement. That
calibration-transaction rung did not test automatic stale-route demotion;
unrestricted memory growth, arbitrary new computation, and general continual
learning also remain unqualified. Evidence is in
`session_records/brainworkshop_canonical_2026-08-05/context_route_reversal_seed17.json`,
`...seed18.json`, and `...seed19.json`.

## Failure-only stale-route demotion (2026-08-05)

The route ledger now has an end-to-end failure-only reversal rung. Cue 4 was
first mastered by n-back-2 in slot zero. A new n-back-4 capability was then
learned and calibrated under cue 5 in slot one. The benchmark subsequently
changed the task behind cue 4 to n-back-4 without any new cue-4 calibration;
only fresh scalar verifier outcomes from those changed lifetimes were allowed
to update the cue-4 route.

`PersistentOpaqueContextRouteEvidence.observe_batch()` groups attempted
context/slot outcomes before advancing the route ledger. This makes reversal
patience a property of fresh rollout batches rather than raw eligible-trial
count, while preserving the learner-visible boundary of learned event tensors,
opaque attempted slots, and deterministic scalar outcomes.

Across seeds 17, 18, and 19, the protected old route was demoted after four
low grouped observations, the previously learned replacement became
preferred, fresh changed-task accuracy was `1.0000`, and the old forced-slot
capability retained `1.0000`. Route-state reload selected the replacement;
controller and prior-bank hashes were unchanged; and replay was zero. This
promotes bounded failure-driven nonstationary external memory: a frozen
controller can revise route policy from new evidence without deleting old
capability state. It does not establish unrestricted memory growth, arbitrary
new computation, or general continual learning. Evidence is in
`session_records/brainworkshop_canonical_2026-08-05/context_route_failure_demotion_seed17.json`,
`...seed18.json`, and `...seed19.json`.

## Generic adaptive capability growth (2026-08-05)

The next growth rung removes the benchmark horizon from appended capability
provisioning. The old compatibility slot learns n-back-2. Two new slots are
then created with the same `AdaptiveOnlineEpisodicRelationReader` and the
same fixed event-window capacity of five. Their capability constructors
receive no n-back value; n-back-3 and n-back-4 exist only inside the private
verifier harness. Each slot learns from fresh scalar outcomes and then passes
an explicit forced candidate audit that calibrates its rendered event cue.

The adaptive reader scores each present event/action/outcome row before
mixing relation contexts. This preserves candidate-specific relations when a
generic window contains extra history, unlike the older pooled reader, which
blurred some horizons under the same capacity.

Across seeds 17, 18, and 19, fresh n-back-2/3/4 accuracy was
`1.0000/1.0000/1.0000`; time-shuffle controls were `0.482--0.560`,
history-reset controls `0.500--0.600`, cue-shuffled controls separated by the
registered margin, and every candidate audit was `1.0000`. Old forced-slot
retention stayed `1.0000`; route reload restored slots `0/1/2`; controller and
prior-bank hashes were unchanged; and replay was zero. This promotes bounded
generic external capability growth, not arbitrary program induction,
unrestricted memory growth, or general continual learning. Evidence is in
`session_records/brainworkshop_canonical_2026-08-05/adaptive_capability_growth_seed17.json`,
`...seed18.json`, and `...seed19.json`.

## Automatic route discovery without new-cue calibration (2026-08-05)

The explicit candidate-calibration bottleneck is now removed for the bounded
cue-conditioned case. After a generic n-back-3 capability was acquired under
cue 5, no forced cue-5 candidate audit wrote route evidence. Twelve fresh
failure-gated fallback batches alone updated the cue-5 context ledger through
grouped scalar outcomes.

Across seeds 17, 18, and 19, the new cue record accumulated eight low slot-0
observations and twelve successful slot-1 observations. Slot 1 became
protected and preferred automatically; fresh cue-5 accuracy was `1.0000`, old
forced n-back-2 retention was `1.0000`, route reload selected slot 1, the
controller and prior-bank hashes were unchanged, and replay was zero. The
new-cue calibration flag was false in every report.

This promotes bounded automatic route discovery from ordinary verifier
outcomes. It still depends on an observable cue, a pre-existing generic
capability blueprint, and stable evidence. Arbitrary new computation,
unrestricted memory growth, and general continual learning remain open.
Evidence is in
`session_records/brainworkshop_canonical_2026-08-05/automatic_route_discovery_seed17.json`,
`...seed18.json`, and `...seed19.json`.

## Failure-triggered adaptive capacity growth (2026-08-05)

The next growth boundary makes external capacity expansion conditional on
fresh opaque failure rather than on benchmark-horizon provisioning. A generic
`AdaptiveOnlineEpisodicRelationReader` starts at event-window capacity `5`.
When a new candidate fails the registered stable mastery threshold of `0.8`
under fresh scalar outcome probes, only that unmastered candidate may be
replaced by the same reader blueprint at capacity `6`. The controller, event
frontend, mastered slots, and prior route state remain unchanged, and the
transaction uses zero replay.

Across seeds 17, 18, and 19, the failure trigger fired, the expanded n-back-6
candidate reached `1.0000` fresh accuracy, ordinary fallback outcomes
discovered its cue-conditioned route, and old n-back-2 retention remained
`1.0000`. Time-shuffle controls were `0.487--0.518`, history-reset controls
were `0.500`, route-state reload selected the new slot, controller and prior
bank hashes were unchanged, and no new cue-calibration transaction occurred.

This promotes bounded failure-triggered external capacity growth. The reset
permission is explicitly limited to an unmastered candidate; it is not a
general optimizer for mastered memory. Unrestricted memory growth, arbitrary
new computation, and general continual learning remain unqualified. Evidence
is in
`session_records/brainworkshop_canonical_2026-08-05/capacity_growth_from_failure_seed17.json`,
`...seed18.json`, and `...seed19.json`.

## Recursive failure-triggered capacity growth (2026-08-05)

The single expansion boundary now composes across two generations. A first
unmastered adaptive capability grows from event-window capacity `5` to `6`;
after it is mastered and routed, a second unmastered capability grows from
`6` to `7`. A failed candidate is replaced as a complete external slot,
including its intention adapter, route scorer, opaque capability key, and
keypress decoder. This is required because reader-only reset can preserve
contaminated output state, as the seed-18 diagnostic exposed. The controller
and every mastered prior slot remain frozen and isolated.

Across seeds 17, 18, and 19, both fresh failure triggers fired, both expanded
capabilities reached `1.0000`, the old n-back-2 and first n-back-6 capabilities
retained `1.0000`, cue-5 and cue-6 routes were discovered from ordinary
fallback outcomes, and all causal controls passed. Reload selected cue routes
`0/1/2`, prior mastered state was unchanged, the controller stayed frozen, and
replay was zero. Each seed used `255,108` unique verifier bits, `53,990`
logical lifetimes, `1,600` optimizer updates, `255,108` eligible feedback
events, and `647,880` total verifier outcome events.

This promotes recursive bounded failure-triggered external capacity growth.
It does not establish unbounded memory growth, learned consolidation or
eviction, arbitrary new computation, or general continual learning. Evidence
is in
`session_records/brainworkshop_canonical_2026-08-05/recursive_capacity_growth_seed17.json`,
`...seed18.json`, and `...seed19.json`.

## Retention-safe bounded eviction and replacement (2026-08-05)

The external capability bank now has a verified bounded lifecycle transaction,
not only append-only growth. In a three-slot live bank, two mastered
capabilities are protected while fresh opaque verifier outcomes identify an
unmastered candidate for replacement. `CapabilityRetentionLedger` masks
protected rows; slot reuse resets the complete external capability and clears
both global and context-conditioned route evidence before a new capability is
trained.

Across seeds 17, 18, and 19, fresh incoming-task failure accuracy was
`0.645/0.655/0.592`, replacement mastery was `1.000` in every seed, and both
prior mastered capabilities retained `1.000`. Stale routes were cleared,
replacement routes were rediscovered from ordinary outcomes, reload selected
`0/1/2`, a fully protected bank refused eviction, the controller remained
frozen, prior mastered state was unchanged, and replay was zero. Each seed
used `260,968` unique verifier bits, `47,846` logical lifetimes, `1,408`
optimizer updates, `260,968` eligible feedback events, and `621,998` total
verifier outcome events.

This promotes retention-safe bounded eviction and slot reuse. The current
eviction utility is an opaque outcome-derived score combined with an explicit
mastery-protection mask; learned general utility, persistent consolidation,
unbounded memory growth, arbitrary new computation, and general continual
learning remain unqualified. Evidence is in
`session_records/brainworkshop_canonical_2026-08-05/protected_eviction_growth_seed17.json`,
`...seed18.json`, and `...seed19.json`.

## Learned context-conditioned capability utility (2026-08-05)

The next boundary makes the eviction score genuinely learned without exposing
the verifier's outcome history to the selector. `ExternalCapabilityEvictionPolicy`
receives one incoming learned event tensor and detached opaque capability
addresses. Fresh scalar verifier outcomes are used only by the external policy
trainer to provide paired utility signals. Candidate capability identities and
physical slots are independently permuted, while incoming context alternates
between the two candidates, so slot order and a hand-written outcome threshold
cannot solve the task.

Across seeds 17, 18, and 19, learned selection was `1.000` in every run;
reward-shuffled and corrupted-feature controls were `0.500` in every run;
replacement fresh accuracy was `0.919/1.000/1.000`; retained capability and
base retention were `1.000` in every run; stale route evidence was cleared; the
controller was frozen; and replay was zero. Each run used `945,792` unique
verifier bits, `236,288` logical lifetimes, `2,912` optimizer updates, and
`2,540,544` verifier outcome events. The stable replacement schedule uses a
`3e-3` learning rate; the rejected `1e-2` schedule passed selection but failed
fresh replacement on two seeds.

This promotes learned context-conditioned utility for a bounded opaque
capability bank, with retention masking still explicit and outside the learned
selector. It does not establish persistent consolidation, unrestricted memory
growth, arbitrary new computation, or general continual learning. Evidence is
in
`session_records/brainworkshop_canonical_2026-08-05/learned_eviction_context_seed17.json`,
`...seed18.json`, and `...seed19.json`.

## Replicated two-step isolated-credit growth (2026-08-05)

The episodic boundary now survives two sequential fresh additions. After the
old context and route are frozen, each new capability receives its own
replaceable `EpisodicCreditHead` and route extension. A later procedure first
passes through the old route and the earlier extension; the earlier extension
and its credit state remain frozen.

Across seeds 69316 and 69317, both new routes selected at `1.000/1.000`, old
route retention stayed intact, prior-extension attempts were present, and
isolated old/new credit-position accuracy was `1.000/1.000`. New-artifact
ablations and reward-shuffled extension controls selected neither new route;
all updates after each append used zero replay.

This promotes bounded two-step external growth with isolated credit state. It
does not establish unrestricted memory growth, learned eviction,
nonstationary discovery, arbitrary program induction, or general continual
learning. Evidence is in
`session_records/sequence_working_memory_2026-08-02/episodic_context_credit_multistep_v1_2026-08-05/`.

## Four-step replay-free isolated-credit growth (2026-08-05)

The same frozen episodic context and route boundary now composes four
sequential additions. Families `2,3,4,5` receive independent replaceable route
extensions and event-credit heads. A later family is evaluated only after the
old route and every earlier extension have failed; future inactive extensions
are excluded from the prior-attempt gate. Credit evaluation compares opaque
temporal positions modulo episode length, so family identifiers cannot be
mistaken for event positions.

Across seeds 69316 and 69317, old and new route selection, candidate
permutation, old-route retention, and isolated credit accuracy were all
`1.000`. Every required prior extension was attempted at `1.000`; disabling
the required extension reduced selection to `0.000`; reward-shuffled
extensions selected at `0.000`; and replay remained zero. Each seed used
`122,880` unique verifier bits, `30,976` logical lifetimes, and `2,048`
optimizer updates.

This promotes bounded four-step replay-free external growth with isolated
episodic credit state. It does not establish unbounded memory growth, learned
consolidation, arbitrary program induction, or general continual learning.
Evidence is in
`session_records/sequence_working_memory_2026-08-02/episodic_context_credit_four_step_v1_2026-08-05/`.

## Eight-step replay-free episodic-context growth (2026-08-05)

The four-step result was extended beyond its finite four-token pattern bank.
Ten same-statistics temporal patterns of length five provide old families
`0,1` and eight sequential additions `2..9`. The context encoder and old
router are frozen before acquisition; each new family receives an isolated
route extension and event-credit head, and later families must attempt all
earlier extensions before activation.

Across seeds 69316 and 69317, old-route accuracy, pooled-baseline separation,
candidate permutation, all eight new routes, old-route retention, and isolated
old/new credit accuracy were `1.000`. Required prior extensions were attempted
at `1.000`; disabling each required extension reduced selection to `0.000`;
reward-shuffled extensions selected at `0.000`; and replay remained zero. Each
seed used `286,720` unique verifier bits, `62,464` logical lifetimes, and
`4,352` optimizer updates.

The short-budget control failed old-route retention at `0.500` on both seeds.
The promoted schedule therefore increases frozen-context and route training
for the longer episode while keeping each new external credit head at 128
fresh updates. This qualifies bounded eight-step replay-free external growth
with isolated episodic credit state, not unbounded memory growth, learned
consolidation, arbitrary program induction, or general continual learning.
Evidence is in
`session_records/sequence_working_memory_2026-08-02/episodic_context_credit_eight_step_v1_2026-08-05/`.

## Generated-pattern length-six replay-free growth (2026-08-05)

The temporal procedure bank is now generated from episode length instead of
being fixed to six hand-listed patterns. At length six, all 20 binary
patterns with three active positions have identical single-event statistics.
After the old context and router were frozen, families `2..9` were acquired
sequentially with isolated route and event-credit state.

Across seeds 69316 and 69317, old-route accuracy, candidate permutation, all
eight new routes, old-route retention, and isolated credit accuracy passed;
required prior extensions were attempted at `1.000`; disabling each required
extension and reward-shuffling selected at `0.000`; and replay remained zero.
Each seed used `393,216` unique verifier bits, `75,776` logical lifetimes, and
`5,632` optimizer updates. The lower context/router budget was retained as a
rejected control because seed 69316 collapsed the old route to `0.500`.

This promotes generated-pattern bounded eight-step replay-free external
growth and removes the fixed pattern-bank ceiling. It still does not
establish unbounded memory growth, learned consolidation, arbitrary program
induction, or general continual learning. Evidence is in
`session_records/sequence_working_memory_2026-08-02/episodic_context_credit_generated_len6_eight_step_v1_2026-08-05/`.

## Retention-safe generated length-six growth (2026-08-05)

The generated length-six sequence is now composed with the external retention
ledger rather than treating growth as append-only. Across seeds 69316 and
69317, ten opaque capabilities initially became protected; a fully protected
bank refused eviction; four sustained low verifier outcomes released only the
newest capability; and four fresh successful outcomes re-protected it. Route,
permutation, causal-ablation, isolated-credit, reward-shuffle, and zero-replay
gates remained passing.

Each seed used `393,304` unique verifier bits, `75,864` logical lifetimes,
`5,632` optimizer updates, and `88` retention observations. This promotes a
bounded replay-free growth contract with a reversible retention boundary. It
does not establish learned consolidation, unrestricted memory growth,
arbitrary new computation, or general continual learning. Evidence is in
`session_records/sequence_working_memory_2026-08-02/episodic_context_credit_generated_len6_eight_step_retention_v1_2026-08-05/`.

## Retention-aware artifact consolidation (2026-08-05)

The memory boundary now prevents generic compaction from silently dropping a
protected artifact. Consolidating protected source rows requires fresh opaque
candidate outcomes; an accepted replacement records those outcomes in the
persisted retention ledger. A two-phase audit first verifies a candidate
without adoption, then protects both source rows from eight fresh probes before
the final one-row consolidation.

Across seeds 69316 and 69317, aliases and executable views survived reload,
behavior and the frozen controller core were preserved, checksum corruption was
rejected, and the rejected-candidate control was not adopted. Each seed used
zero consolidation optimizer updates and zero replay. The short 64-update
control failed stable candidate mastery and was rejected.

This promotes retention-aware behavior-verified logical compaction. It does not
establish learned byte compression, unrestricted memory growth, arbitrary new
computation, or general continual learning. Evidence is in
`session_records/sequence_working_memory_2026-08-02/artifact_consolidation_retention_v2_2026-08-05/`.

## Opaque learned consolidation proposal boundary (2026-08-06)

`OpaqueConsolidationPolicy` is now a canonical replaceable memory-side
component. It consumes only controller-native keys, values, strength, and
relative age; pair features are symmetric under candidate permutation. The
policy proposes a mechanical merge or keep operation, while
`verify_consolidation_proposal` rewrites an immutable tensor snapshot and
requires an independent verifier and optional retention gate before adoption.

Across seeds 69316 and 69317, the policy selected a verifiable pair on every
512-bank held-out audit, preserved candidate permutation, beat untrained and
reward-shuffled controls, and composed four retention-gated rewrites from eight
rows to four with zero replay. This promotes learned opaque rewrite selection
and sequential latent compaction. It does not yet establish learned
executable-artifact behavioral consolidation, learned byte compression,
unrestricted memory growth, arbitrary new computation, or general continual
learning. Evidence is in
`session_records/sequence_working_memory_2026-08-02/opaque_consolidation_v1_2026-08-06/`.

## Behavior-preserving migration order

1. ~~Extract the existing vision encoder and actuator from controller ownership.~~
2. ~~Add `encode`, `step_events`, and `decode` boundaries.~~
3. ~~Prove bit-identical behavior and checkpoint conversion for the current path.~~
4. ~~Version the event and intention schemas and save adapters separately.~~
5. ~~Remove the active legacy action residual by folding it into the base
   intention without losing any admitted capability.~~
6. ~~Add a second output adapter and prove simultaneous M-output fan-out from
   the clean base intention.~~
7. ~~Accept variable-size synchronous visual event sets.~~
8. ~~Qualify timestamp-preserving out-of-order and bounded-jitter delivery.~~
   A generic timestamp window buffer now also restores one-step delayed audio
   composition with one step of measured arrival latency; evidence is in
   `session_records/amodal_latent_alignment_2026-08-03/`. The same generic
   buffer now has a bounded timeout that releases an explicit partial window
   with presence masking when a stream is absent; redundant-view audits retain
   89.26--99.49% accuracy with exactly one partial window and no stuck pending
   state. A payload-blind arrival predictor now learns wait/proceed timing from
   presence, age, and arrival history: two 128-lifetime audits match a fixed
   two-step timeout within two points of accuracy and 0.01 verified utility
   while reducing query latency by 0.04--0.08 event units. This is learned
   timing; a verifier-only six-way utility grid selected its wait threshold on
   a non-held-out split, and two fresh held-out audits passed. It is not
   recovery of genuinely absent complementary evidence.
9. ~~Add a runtime-variable encoder registry and controller/decoder wrapper.~~
   A reverse-basis synthetic encoder can now be aligned with a frozen controller
   by paired unlabeled sensory consistency; the three-appearance composition and
   causal controls are recorded in
   `session_records/amodal_latent_alignment_2026-08-03/`. This is a qualified
   neural-IR alignment diagnostic, not arbitrary natural-modality transfer.
   The saved adapter replay is independently qualified, and a synthetic raw
   audio frontend now passes the same frozen-core composition and causal gates.
   Waveform noise and burst-dropout robustness are also qualified. A separate
   continuous RGB patch-token frontend now passes the same frozen-core
   composition and causal gates from paired unlabeled consistency; independent
   replay is recorded alongside the audio reports. A discrete text-like sensor
   now passes the same gates as well, using an embedding frontend and 768
   meaningless symbols per frame. Natural-language semantics, arbitrary raw
   token streams, and cold-start alignment remain open.
10. Require complementary evidence split across two encoders.
11. Train audio and language frontends/backends only after the generic buses pass.
12. ~~Define and qualify the replaceable `MemoryBackend` boundary.~~ The v1
    in-process and atomic disk-backed implementations round-trip through
    runtime checkpoints. Scalar outcome recall is promoted through a
    differentiable training transaction, and v18 promotes fixed-write
    two-row content binding across isolated batch scopes. Commit rates varied
    in the scalar rung, and the first v20 cue-guided retention rung was
    rejected, so learned skip or utility-based retention remains unqualified,
    as do persistent episodic utility and cross-adapter retrieval.
13. Freeze the controller and qualify genuinely new sensors and outputs.

No capability claim may skip from the current internal vision module directly
to “amodal.” Each boundary must pass its own causal and sample-efficiency gate.

## Retention-safe online executable growth (2026-08-06)

The canonical external-memory boundary now supports a retention-gated online
addition of one executable view. Fresh opaque verifier outcomes protect the
old capabilities before extension; the candidate is built and probed in a
disposable transaction; and independent behavior verification must preserve
the old capabilities before the replacement is committed. The promoted audit
keeps the controller and old router frozen, learns the new route from fresh
outcomes with zero replay, and persists five opaque views in one physical row.

This is an important continual-learning safety primitive, but the claim is
bounded: it covers one new view, finite memory, and an externally trained
extension. It does not yet demonstrate unrestricted memory growth, arbitrary
new computation, learned byte compression, or general continual learning.
Evidence is in
`session_records/sequence_working_memory_2026-08-02/online_view_growth_retention_v2_2026-08-06/`.

## Composed retention-safe growth (2026-08-06)

The retention transaction composes across two sequential additions. The first
replacement must establish protected mastery before the second candidate is
constructed; each candidate is independently probed and behavior-verified;
and the final memory row preserves six opaque executable views while the
controller and old route remain frozen. The promoted two-seed audit uses zero
replay after either addition and rejects a short-budget control at the first
retention failure.

This qualifies composition of a bounded external-memory safety primitive. It
does not yet qualify open-ended growth, finite-capacity consolidation under
pressure, arbitrary new computation, or general continual learning. Evidence
is in
`session_records/sequence_working_memory_2026-08-02/multistep_view_growth_retention_v2_2026-08-06/`.

## Protected capacity growth (2026-08-06)

The executable artifact boundary now has an explicit capacity-growth
transaction. When all rows are protected, a write refuses eviction. The
caller can create a larger verified store that copies artifacts, opaque
aliases, strengths, and retention records while leaving the source immutable,
then admit the new capability. This prevents “unbounded growth” from being
implemented as silent forgetting, but it does not yet provide learned capacity
planning, learned compression, or general continual learning. Evidence is in
`session_records/sequence_working_memory_2026-08-02/artifact_capacity_growth_v1_2026-08-06/`.

## Longer growth exposes the representation bottleneck (2026-08-06)

The three-step pressure test composes seven opaque executable views and
requires retention-safe float16, int8, and int4 replacements. Two independent
seeds now pass the full boundary with zero replay and frozen controller and
earlier extensions. A historical rejection exposed inconsistent raw-minimum
versus stable-prefix retention accounting; the promoted implementation now
uses the stable-prefix definition consistently and records a paired
full-precision source control. The result is still bounded continual growth,
not unrestricted memory growth or general continual learning. Evidence is in
`session_records/sequence_working_memory_2026-08-02/three_step_view_growth_retention_exploration_2026-08-06/`.

## Learned capacity admission planning (2026-08-06)

The next memory bottleneck was explicit caller-side capacity choice. The
package now exposes `OpaqueCapacityPlanner`, a replaceable memory-side policy
that scores four generic actions: admit into free capacity, evict an
unprotected row, consolidate a pair, or grow the bank. It consumes only an
incoming learned key/value, fixed-width opaque artifact summaries, generic
strength/age metadata, and explicit protection/transaction availability. Its
row and pair scores are permutation-equivariant; it does not receive task
labels, modality data, or protocol fields.

Across two seeds, the learned action accuracy was `0.976--0.980`, the
ambiguous eviction-versus-consolidation choice was `0.879--0.909`, and the
reward-shuffled ambiguous control was `0.545` (near the two-choice floor).
When every executable row was protected and verified consolidation was
unavailable, the planner selected growth. The source bank remained immutable,
retention transferred, the new artifact was admitted, and all artifacts
reloaded successfully with zero replay and zero controller updates.

This promotes bounded learned admission planning and makes capacity growth a
first-class replaceable memory decision. Protection masking, executable
behavior verification, and transaction adoption remain explicit safety gates.
It does not establish learned consolidation of arbitrary procedures,
unrestricted memory growth, arbitrary new computation, or general continual
learning. Evidence is in
`session_records/sequence_working_memory_2026-08-02/artifact_capacity_planning_v1_2026-08-06/`.

## Twelve-addition retention scaling rejection (2026-08-06)

The generated length-six episodic pressure test was extended from eight to
twelve sequential additions, producing fourteen opaque capabilities. The
context, route, isolated-credit, and causal mechanisms continue to work, but
the cross-seed retention gate does not: seed 69316 passes while seed 69317
leaves the final capability at `0.8125` fresh route selection and fails stable
initial protection. The twelve-addition rung is therefore rejected; the
eight-addition result remains the highest promoted episodic-retention boundary.

This localizes the next bottleneck to confidence-aware retention calibration
and route margin under a larger candidate bank. Lowering the mastery threshold
or excluding the final capability would hide the failure rather than solve it.
Evidence is in
`session_records/sequence_working_memory_2026-08-02/episodic_context_credit_generated_len6_twelve_step_retention_rejected_v1_2026-08-06/`.

The intermediate ten-addition rung passes both seeds with the original
budgets. It is now the highest promoted generated length-six boundary: twelve
opaque capabilities retain their routes and isolated credit state, and the
fully protected bank still refuses eviction before the final capability is
reversed and recovered. Evidence is in
`session_records/sequence_working_memory_2026-08-02/episodic_context_credit_generated_len6_ten_step_retention_v1_2026-08-06/`.

A twelve-addition control with every extension budget doubled to 256 updates
also failed cross-seed retention: seed 69317 fell to `0.75` on the final
route. This rejects longer per-extension training as a sufficient repair and
keeps route-interference/confidence calibration as the next target. Evidence
is in
`session_records/sequence_working_memory_2026-08-02/episodic_context_credit_generated_len6_twelve_step_ext256_rejected_v1_2026-08-06/`.

The final calibrated twelve-addition rung now passes both seeds. Each new
extension receives fresh positive outcomes for its own context and fresh
negative outcomes for already-acquired contexts, and its loss is aligned to
the `>1.0` activation boundary. This removes the earlier route-interference
and score-calibration failure without changing the retention gates. Evidence
is in
`session_records/sequence_working_memory_2026-08-02/episodic_context_credit_generated_len6_twelve_step_calibrated_v1_2026-08-06/`.

## Full generated pattern-bank growth (2026-08-06)

The calibrated growth boundary now fills the entire generated length-six
pattern bank: two frozen old capabilities plus 18 sequential additions, or
20 opaque capabilities total. Across seeds 69316 and 69317, the minimum new
route selection is `0.875` and `0.8125`; old-route and candidate-permutation
accuracy are `1.000` for both. Full-bank eviction refusal, single-target
reversal, fresh recovery, isolated credit, balanced reward-shuffle, causal
extension, and zero-replay gates all pass.

This is the highest promoted bounded retention result in the current family.
It does not establish unbounded growth: the pattern family is still a closed
20-capability bank and the route/credit machinery is externally trained. The
next meaningful pressure test is a changing distribution or a dynamically
expanded store, not more updates on the same closed bank. Evidence is in
`session_records/sequence_working_memory_2026-08-02/episodic_context_credit_generated_len6_full_bank_v1_2026-08-06/`.

## Nonstationary temporal shift exposes credit calibration (2026-08-06)

The next pressure test freezes two capabilities acquired on length-six
episodes, then appends eight new routes and isolated credit heads from fresh
length-seven episodes. The old stream is not replayed after the shift. Across
both seeds, old-route retention, candidate permutation, new-route selection,
shifted credit localization, reversal/recovery, and zero replay pass. The
minimum new-route selection is `0.9688/0.9375`.

The rung is rejected because seed 69316 activates shifted family 7 at `0.6406`
under reward-shuffled outcomes, while seed 69317 remains clean. This is a
real nonstationary credit-calibration failure, not a storage or retention
failure. The audit therefore remains rejected until shifted negative credit is
robust across seeds. Evidence is in
`session_records/sequence_working_memory_2026-08-02/nonstationary_growth_episode6_to7_rejected_v1_2026-08-06/`.

The repair uses an antithetic shuffled null: each shifted query is duplicated
with contradictory scalar outcomes, making null credit exactly zero while
leaving aligned acquisition unchanged. The corrected two-seed rung passes all
gates, including the previously failing negative control; minimum new-route
selection is `0.9688/0.9375`, shifted credit is `1.000` for old, new, and
combined positions, and replay remains zero. This promotes one controlled
temporal shift, not arbitrary nonstationarity or general continual learning.
Evidence is in
`session_records/sequence_working_memory_2026-08-02/nonstationary_growth_episode6_to7_v2_2026-08-06/`.

The combined full-bank follow-up also passes both seeds: 18 fresh length-seven
capabilities fill the 20-family bank after the length-six base, with minimum
new-route selection `0.9688/0.9219`, perfect old-route and permutation gates,
shifted credit at `1.000/1.000` and `0.9444/0.9500` for old/new/combined
positions, isolated reversal/recovery, and zero replay. This is the strongest
bounded result so far, but it remains one controlled shift over a closed
family bank. Evidence is in
`session_records/sequence_working_memory_2026-08-02/nonstationary_growth_full_bank_v1_2026-08-06/`.

Increasing the shift magnitude to length eight also passes both seeds. The
minimum new-route selection is `0.8906/0.8281`; old-route and permutation
accuracy remain perfect, shifted credit remains `1.000` for old, new, and
combined positions, and reversal/recovery, antithetic null, and zero-replay
gates pass. This is evidence against an adjacent-length shortcut, but it is
still one controlled shift over a closed bank rather than general continual
learning. Evidence is in
`session_records/sequence_working_memory_2026-08-02/nonstationary_growth_full_bank_6to8_v1_2026-08-06/`.

The larger length-six→length-ten shift exposes the next retention boundary.
Seed 69316 passes, but seed 69317 leaves shifted family 12 below the mastery
threshold before reversal; the full protected bank therefore fails to refuse
eviction. Routing, causal credit, antithetic null, and zero-replay controls
still pass. The rung is rejected rather than promoted with a lower retention
threshold. Evidence is in
`session_records/sequence_working_memory_2026-08-02/nonstationary_growth_full_bank_6to10_rejected_v1_2026-08-06/`.

The repair doubles new-extension acquisition to 256 updates per family and
zero-centers the antithetic null objective. The corrected 6→10 rung passes both
seeds without changing the mastery threshold: minimum new-route selection is
`0.8438/0.8750`, the full bank protects and recovers correctly, and replay is
zero. The earlier 128-update failure remains the acquisition-depth regression
control. Evidence is in
`session_records/sequence_working_memory_2026-08-02/nonstationary_growth_full_bank_6to10_v2_2026-08-06/`.

The next audit performs two sequential shifts in one frozen run: length six
→ length eight → length ten. Eight capabilities are acquired in the first
wave and ten in the second, with earlier route and credit state untouched.
Across both seeds, phase-one minimum route selection is `0.9219/0.8906`,
phase-two is `0.8906/0.9063`, old-route and permutation gates are perfect,
all-shift credit and causal gates pass, the full bank remains protected, and
replay is zero. This is the first multi-phase continual-growth result in the
current system. It remains bounded by the 20-family bank and does not qualify
general continual learning. Evidence is in
`session_records/sequence_working_memory_2026-08-02/repeated_shift_growth_6to8to10_v1_2026-08-06/`.

The follow-up crosses the former closed-bank ceiling in one frozen run:
length six → length eight → length ten → length twelve. Eight, ten, and twelve
new capabilities are acquired in sequence, producing a 32-capability bank.
Across both seeds, phase minima are `0.9219/0.8906`, `0.8906/0.9063`, and
`0.9219/0.8594`; old-route/permutation, causal credit, full-bank protection,
isolated reversal/recovery, antithetic null, and zero-replay gates all pass.
This promotes dynamic external-bank growth beyond 20 capabilities, but not
unbounded learned expansion: the generated family bank remains finite and the
route/credit acquisition machinery remains externally trained. Evidence is in
`session_records/sequence_working_memory_2026-08-02/repeated_shift_growth_6to8to10to12_v1_2026-08-06/`.

The external growth boundary now also exposes `ExternalGrowthPrior`: a
copy-on-write average of independently acquired adapter state. On the same
32-capability schedule, both seeds preserve route, causal credit, retention,
reversal, null-control, and zero-replay gates when later adapters use the
prior. The result qualifies safe external-state reuse, but not a reliable
transfer gain; a matched 128-update prior control fails both seeds in the
final shift. Late-shift acquisition depth and learned prior calibration
therefore remain open. Evidence is in
`session_records/sequence_working_memory_2026-08-02/repeated_shift_growth_6to8to10to12_growth_prior_v1_2026-08-06/`.

## Shared variable-candidate growth router (2026-08-06)

The next implementation step removes the one-extension-per-capability
assumption. `OpaqueCandidateGrowthRouter` is one permutation-equivariant,
candidate-conditioned scorer for each shift. It accepts a variable opaque
candidate bank and a learned query summary, so the number of external
capabilities can grow without adding a modality- or capability-specific
reasoning branch to the controller. The promoted query summary concatenates
the learned context with final, mean, and max recurrent states; candidate keys
remain random opaque vectors.

Across two seeds, one shared router adds 8, 10, and 12 capabilities in
successive length-six → length-eight → length-ten → length-twelve shifts.
The 32-row bank reaches phase minima of `0.9844/0.9375`,
`0.9844/0.9688`, and `0.9531/0.9375`. Direct old/new candidate permutation,
causal credit, full-bank protection, isolated reversal/recovery,
reward-shuffled null, and zero-replay gates all pass. This promotes a
reusable variable-bank growth mechanism and is stronger architecturally than
the prior bank of per-capability extensions.

The result is deliberately not overstated. The corrected strict sequential
operational route-permutation diagnostic is `0.9906/0.9911`, matching the
direct candidate-score audit; the earlier `0.4932/0.4943` result was a harness
false negative caused by comparing a remapped physical row to its unpermuted
family index. Route acquisition still uses 16,384 updates per shared
expansion, so this is not yet a sample-efficiency gain. The next pressure test
is to reduce acquisition cost and remove the fixed trajectory summary.
Evidence is in
`session_records/sequence_working_memory_2026-08-02/shared_growth_router_6to8to10to12_trajectory_stats_v1_2026-08-06/`.

## Shared-router acquisition efficiency (2026-08-06)

The same shared-router architecture now passes the complete 6→8→10→12
schedule with 8,192 rather than 16,384 updates per expansion. Across two
seeds, phase minima are `0.9844/0.9844`, `0.9688/0.9063`, and
`0.9219/0.9063`; direct candidate permutation is exact and operational
permutation is `0.9875/0.9802`. Causal credit, full-bank protection,
reversal/recovery, reward-shuffled null, and zero replay all pass. Total
optimizer updates fall from `104,704` to `55,552` per seed, a `46.9%`
reduction, while the bank still reaches 32 capabilities.

This promotes the reduced acquisition budget, not the exploratory prior or
prototype-address controls: those did not replicate across seeds. The fixed
trajectory-statistics query and arbitrary random candidate-key associations
remain the next generalization bottleneck. Evidence is in
`session_records/sequence_working_memory_2026-08-02/shared_growth_router_6to8to10to12_trajectory_stats_8192_v1_2026-08-06/`.

## Fourth-shift capacity frontier rejected (2026-08-06)

The next 6→8→10→12→14 audit grows the bank to 46 capabilities. The
8,192-update hidden-256 control reaches a final route floor of `0.8125`, but
two late rows fail stable protection. A late-shift budget increase to 12,288
updates worsens the floor to `0.7188`; hidden size 512 at the original budget
reaches `0.7969` and also fails. Old-route retention, permutation, causal
credit, reward-shuffled null, and zero replay remain intact in all controls.

This rejects naive width and budget scaling at the 46-capability frontier.
The next implementation target is confidence-aware late-shift acquisition and
capacity planning that can detect under-mastered rows, allocate targeted fresh
outcomes, and refuse promotion until the full bank is protected. Evidence is
in
`session_records/sequence_working_memory_2026-08-02/shared_growth_router_6to8to10to12to14_46caps_rejected_v1_2026-08-06/`.

## Frozen-core recurrent transfer control (2026-08-06)

The next bottleneck was tested directly rather than inferred from retention:
does reusing a mastered controller make acquisition of a new procedure faster
than a fresh learner? The matched harness gives both arms the same fresh
span-four reverse episodes and the same update budget. The inherited arm keeps
the controller core frozen and trains only an opaque recurrent growth slot;
the fresh arm is deliberately stronger, with all parameters trainable. Fresh
span-two parent-task outcomes are interleaved as online retention rehearsal,
not replayed examples.

The recurrent slot plus fresh rehearsal preserves the parent and the frozen
core. Seed 69316 reaches stable target mastery at `6,144` verifier bits versus
`9,216` for the fresh learner (`1.50x` fresh-over-transferred). Seed 69317
passes target, retention, core-immutability, and reward-shuffled gates but
ties the fresh learner at `12,288` bits. Width, stronger-parent, and learned
output-gate controls do not remove the tie, so the population transfer gate
is not promoted.

This qualifies a narrow retention-safe recurrent external-growth diagnostic,
not a replicated sample-efficiency gain. The current blocker is reliable
positive transfer into new computation when the inherited parent is only
moderately mastered; more storage capacity or longer training alone is not
yet evidence of a solution. Evidence, including rejected controls, is in
`session_records/sequence_working_memory_2026-08-02/frozen_core_transfer_recurrent_rehearsal_v1_2026-08-06/`.

## Parent-conditioned recurrent external transfer (2026-08-06)

The transfer bottleneck was context-selective execution. A recurrent growth
slot could acquire a new procedure, but its residual interfered with older
procedures because its output gate saw only its compressed recurrent state.
The canonical optional boundary now lets an external slot consume the frozen
controller's learned intention and lets its generic gate read the current
opaque learned context. Prior-only slots still receive only their preceding
register, and all new paths are zero-output until trained.

The corrected parent-calibrated audit passes two independent seeds. The
inherited learner reaches stable span-four reverse mastery at `9,216` fresh
verifier bits for both seeds, while matched fresh learners require `15,360`
and `12,288` bits, giving `1.667x` and `1.333x` fresh-over-inherited transfer.
Forward and reverse parent primitives remain retained at `1.000`; the frozen
core digest is unchanged; replay is zero. The shuffled arm adds no target gain
over the parent's pre-growth target baseline and the transferred arm exceeds
it by `0.188` and `0.250`.

This promotes a narrow parent-conditioned external-computation transfer
primitive. It does not establish unrestricted memory growth, arbitrary
program induction, broad multimodal transfer, or general continual learning.
The next pressure test is multiple simultaneously stored parent-conditioned
programs with persistent reload and route isolation. Evidence is in
`session_records/sequence_working_memory_2026-08-02/frozen_core_transfer_parent_conditioned_v1_2026-08-06/`.

## Parent-conditioned external capability bank (2026-08-06)

The next pressure test replaces the fixed residual growth slot with a
memory-side `ExternalCapabilityProgram`. Each program owns a recurrent
episodic context state and an intention adapter; output decoding remains a
replaceable capability-local decoder on the intention bus. The shared
controller, frontend, and parent output path remain frozen. Two distinct
programs (`reverse4` and `forward4`) are acquired from fresh rendered events,
opaque actions, and scalar outcomes, then stored as opaque files in one
content-addressed bank.

Across seeds `69316` and `69317`, both programs pass stable-prefix mastery,
opaque learned routing, candidate permutation, reward-shuffled routing,
wrong-program causal separation, parent retention, exact reload, corruption
rejection, and frozen-core digest gates with zero replay. Selected program
accuracy is `0.895/0.973` and `0.934/0.961`; wrong-program accuracy is
`0.566/0.535` and `0.539/0.559`.

This promotes the controller-as-CPU / memory-as-files boundary for two
bounded external programs. It is still not general continual learning:
sequential append pressure, eviction and consolidation under a larger bank,
nonstationary route reversal, and open-ended program composition remain the
next blockers. Evidence is in
`session_records/sequence_working_memory_2026-08-02/parent_conditioned_external_capability_bank_v1_2026-08-06/`.

## Three-program external capability capacity (2026-08-06)

The two-file bank was extended to three independently acquired programs:
`reverse4`, `forward4`, and `complement4`. A permutation-equivariant opaque
router now trains from every target-versus-competitor pair rather than a fixed
two-row schedule, which avoids a cyclic pair bias as the bank grows. The
balanced reward-shuffled control is also exactly half-positive per update, so
route-label imbalance cannot create a false negative-control signal.

Both seeds pass stable-prefix mastery for all three programs, learned route
accuracy, candidate permutation, balanced shuffled-route rejection, causal
wrong-program separation, parent retention, exact reload, corruption
rejection, frozen-core immutability, and zero replay. This promotes a narrow
three-file external-computation capacity result. It does not establish
sequential capacity pressure, learned eviction or consolidation,
nonstationary route reversal, unrestricted memory growth, or general
continual learning. Evidence is in
`session_records/sequence_working_memory_2026-08-02/parent_conditioned_external_capability_bank_v2_2026-08-06/`.

## Sequential protected external-capability append (2026-08-06)

The next bottleneck was exercised at the actual memory boundary. The three
programs above are first filled into a capacity-three bank and each row is
protected after fresh stable-prefix retention probes. A newly acquired
`rotate4` program is then attempted against the full bank. Both seeds reject
the write explicitly because every occupied capability is protected; no old
artifact is silently evicted. `ExecutableArtifactMemory.grow()` copies the
opaque addresses, artifacts, checksums, and retention ledger into capacity
four, after which the new artifact is appended and independently retained.

The old three-row router and frozen parent are not updated after the append.
A separate `OpaqueViewRouteExtension` learns the new route from fresh
operation queries and paired scalar verifier outcomes only. Both seeds pass
old-route, new-route, combined-route, and candidate-permutation accuracy at
`1.000`; selected capability accuracy is `0.918/0.973/0.902/0.945` and
`0.898/0.973/0.984/0.988` for `reverse4/forward4/complement4/rotate4`.
Reload, corruption rejection, frozen digests, causal wrong-artifact
separation, balanced reward-shuffled controls, and zero replay also pass.

One rejected control is important: arbitrary random bank keys caused the
old route to collapse to `0.536` at the same budget. Learned event-derived
opaque address evidence is therefore part of the current route contract; the
router is not a magic decoder for unrelated random identifiers.

This promotes one protected sequential append for a frozen processor. It does
not establish repeated open-ended growth, learned eviction or consolidation
under nonstationarity, route reversal, arbitrary program synthesis, or general
continual learning. Evidence, including the rejected keying control, is in
`session_records/sequence_working_memory_2026-08-02/sequential_external_capability_append_v1_2026-08-06/`.

## Canonical external-capability lifecycle (2026-08-06)

The implementation seam exposed by the append audits is now explicit in the
production package. `ExternalCapabilityLifecycle` is a protocol-agnostic
coordinator over `ExecutableArtifactMemory`, retention state, optional
`OpaqueCapacityPlanner`, capacity growth, and verified consolidation. It
returns auditable admission receipts and refuses to adopt an unverified
rewrite; artifact execution, route semantics, and controller training remain
outside the store.

The coordinator-backed two-step audit reruns the four-view bank followed by
`rotate` and `complement_rotate`. Both seeds pass the full boundary: the
two-step chain is `1.000/0.9974`, candidate-permutation accuracy is
`1.000/0.9974`, both new routes are `1.000`, all six capabilities remain
retention-safe, and replay after either append is zero. The controller, old
router, and first extension remain frozen; reload and checksum-corruption
controls pass; and the antithetic reward-shuffled controls remain at the
non-selection baseline.

This promotes a reusable bounded lifecycle transaction, not general
continual learning. The remaining gap is longer nonstationary growth with
confidence-calibrated retention, verified learned consolidation under genuine
capacity pressure, and positive transfer against a fresh learner. Evidence is
in
`session_records/sequence_working_memory_2026-08-02/multistep_view_growth_lifecycle_v1_2026-08-06/`.

## External capability composition pipeline (2026-08-06)

The production package now exposes `ExternalCapabilityPipeline`, a
variable-length memory-side execution chain. Each program keeps its own
recurrent state outside the controller, and only the learned intention is
passed from one program to the next. An empty pipeline is an exact identity;
adding or replacing programs does not resize the controller or create a
task-specific reasoning branch.

The first behavioral pressure test learned separate `complement4` and
`reverse4` programs, froze them, and trained a fresh decoder on the novel
`complement_reverse4` target. The composed pipeline beat the blank pipeline at
`0.9492` versus `0.5508` on seed `69316` and `0.8828` versus `0.6641` on seed
`69317`. Exact reload, corruption rejection, frozen-core, and shuffled-outcome
controls passed; the second primitive was causal on both seeds. However, the
first primitive was not causal on seed `69317`, and a fully fresh trainable
pipeline result from the first harness is invalid because a `no_grad` scope
blocked its program gradients; it is retained only for provenance. This is
retained as a rejected general-composition diagnostic, not as arbitrary
program induction or fresh-learner transfer. A cheap visibility control
rehydrated the seed-69317
artifact and removed raw events from downstream programs; accuracy fell from
`0.8828` to `0.5195`, showing that the current chain still has a substantial
raw-event shortcut. Evidence is in
`session_records/sequence_working_memory_2026-08-02/external_capability_composition_rejected_v1_2026-08-06/`.

The corrected fresh-gradient rerun repairs the control but does not earn a
two-seed promotion: seed `69316` reaches `2,048` composed bits versus `6,144`
for fresh (`3.0x` fresh-over-composed), while seed `69317` reaches `14,336`
composed bits versus `6,144` for fresh (`0.429x`). The first primitive is also
noncausal on seed `69317`. This makes the current transfer effect
seed-sensitive. The next implementation target is verifier-gated candidate
selection: inherited composition state must beat a fresh baseline on a fresh
probe before it is installed. Evidence is in
`session_records/sequence_working_memory_2026-08-02/external_capability_composition_corrected_rejected_v1_2026-08-06/`.

## Head-only intermediate consumer (2026-08-06)

The stricter follow-up trains a downstream external consumer while the
pipeline hides raw events from every program after the head. On seed `69316`,
the consumer reaches `0.8633` versus `0.5586` for a blank pipeline; zeroing the
head or consumer reduces performance to `0.5859` and `0.5391`, respectively.
This demonstrates that a memory-side consumer can learn a causal computation
from an opaque intermediate rather than merely rereading the sensory event.

It is not yet positive continual-learning transfer: the fresh head-only
pipeline reaches stable mastery in `6,144` verifier bits versus `16,384` for
the inherited consumer. The next target is a replay-free curriculum or
external prior that improves consumer acquisition while preserving this
head-only contract. Evidence is in
`session_records/sequence_working_memory_2026-08-02/external_capability_intermediate_consumer_rejected_v1_2026-08-06/`.

## Lifecycle-backed learned executable compaction (2026-08-06)

The same coordinator now owns the learned executable-artifact consolidation
path rather than only the append-growth path. Four independently acquired
artifacts are reduced from four physical rows to one through three immutable,
retention-gated, behavior-verified rewrites; aliases and executable views
survive every step. The final opaque route learner uses 2,048 fresh paired
outcome updates per arm.

Both seeds pass the complete boundary. Route/permutation accuracy is
`1.000/1.000` and `0.9453/0.9453`; shuffled-route accuracy is
`0.2773/0.2500`; every replacement is protected; reload, corruption, causal
wrong-view, frozen-core, and zero-replay gates pass. The 512-update
seed-69317 route control is rejected at `0.8828`, isolating acquisition depth
as a real bottleneck instead of silently promoting an unstable route.

This closes a major implementation inconsistency in the memory boundary. It
still does not establish arbitrary new computation, open-ended nonstationary
growth, robust positive transfer against a fresh learner, or general
continual learning. Evidence is in
`session_records/sequence_working_memory_2026-08-02/learned_executable_consolidation_lifecycle_v1_2026-08-06/`.

## Lifecycle-backed canonical capability composition (2026-08-06)

The canonical controller growth-register pressure test now uses the isolated
`ExternalCapabilityLifecycle` for producer→consumer composition. The producer
artifact and prior-only consumer artifact are built into a namespaced
replacement row in a separate destination; a fresh runtime reloads the row,
and the lifecycle adopts it only when the held-out behavior verifier passes.
The original two rows remain immutable if verification fails.

Two seeds pass the promoted rung with verifier/final composition accuracy of
`0.7682` and `0.6406`, blank-sequence accuracy of `0.5000` and `0.5052`, and
reward-shuffled accuracy of `0.5938` and `0.4453`. Both reduce the bank from
two rows to one, preserve exact artifact reload and frozen-core equality, and
pass producer-zeroing, prior-read-ablation, missing-evidence, and zero-replay
controls. This is the strongest current evidence that the CPU/filesystem
analogy is implemented as an executable external-state transaction rather
than manual side-by-side loading.

The result remains intentionally narrow: it demonstrates composition of two
learned register capabilities in one working-memory computation, not arbitrary
program induction, unrestricted memory growth, broad transfer, or general
continual learning. Evidence is in
`session_records/sequence_working_memory_2026-08-02/canonical_growth_pressure_lifecycle_composition_v1_2026-08-06/`.

## Confidence-aware staged admission (2026-08-06)

The production memory boundary now includes
`ConfidenceAwareCapabilityStaging`. A new opaque artifact is held outside the
executable bank while the frozen controller continues producing verifier
outcomes. Only a stable-prefix mastery gate can promote it through
`ExternalCapabilityLifecycle`; until then it cannot consume capacity, evict a
protected row, or be mistaken for a retained capability.

On promotion, the staging ledger transfers the accumulated scalar evidence
directly into the executable memory ledger. This preserves the no-replay
contract: admission does not regenerate old episodes or fabricate fresh
outcomes. With an optional staging directory, atomic artifact snapshots,
checksums, and restart recovery preserve pending candidates across process
boundaries. Admission and executable-memory replacement remain separate
transactions, so this does not claim a distributed multi-process commit.

This closes the gap between “the retention policy exists” and “unverified
growth is actually prevented from perturbing the protected bank.” It is a
safety and lifecycle gain, not evidence of better route learning or general
continual learning. The implementation is covered by the lifecycle and
retention test suites. On the fixed 46-capability pressure stream, the same
outcome-only audit admitted `43/46` candidates for seed 69316 and `39/46` for
seed 69317; the remaining `3` and `7` stayed pending, with every occupied row
protected and no pending candidate consuming executable capacity. The route
frontier itself remains rejected because the late-shift mastery gate fails.

## Durable learned route state (2026-08-06)

The external-memory contract now includes `PersistentOpaqueStateStore` for
replaceable learned route and utility policy weights. A route policy is no
longer an experiment-local raw `torch.save(state_dict)` beside the artifact
bank: its state is written atomically with a versioned JSON configuration and
SHA-256 tensor digest, and reload validates keys and shapes before mutating the
replacement module. The canonical parent-conditioned and sequential append
audits now use this store for their route policies and extensions.

This closes a persistence inconsistency at the memory boundary. Artifact files,
retention evidence, and the learned policy that addresses them can now be
reloaded as independently versioned external state. It does not make a route
policy semantically general, provide distributed transactions, or establish
general continual learning; route behavior still requires held-out verifier
and corruption controls.

## Shared external register interpreter (2026-08-07)

The next execution boundary is now explicit in the production package as
`ExternalCapabilityRegisterMachine`. It separates a fixed learned
register-to-register interpreter from variable external instruction data:
each instruction is one opaque persisted vector, and a serial chain applies
those vectors to an external working register. The first read may use the
standardized learned event, opaque feedback, and controller intention; later
instructions receive only the preceding register. The controller and output
protocol remain unchanged, and quiet ticks preserve register state without
fabricating evidence.

This is the CPU/filesystem foundation needed for compositional capability
growth, but it is not yet a universal interpreter or a promoted learned
capability gain. The initial unconstrained nonlinear transition reached
`1.0000` on both primitive instructions but only `0.5000` on their frozen
serial composition, exposing a whole-function latent-code shortcut. The
default is now an explicitly factorized transition. In a fixed-codebook
mechanistic follow-up, separate external decoders retained both primitives at
`1.0000` and a fresh decoder reached `1.0000` on the serial composition. This
was a signal, not a promotion. The subsequent rendered-event audit added a
recurrent external context so every active event is ingested before
register-only execution. Across two seeds, reverse retention was
`0.9844/0.9688`, composition was `0.9844/0.9805`, reward-shuffled composition
was `0.4336/0.2891`, and matched fresh composition was `0.9492/0.8750`; exact
reload and frozen-parent controls passed. This was the composition signal;
the promotion-quality rerun then added stable-prefix bits, missing-evidence,
and checksum-corruption gates. Both seeds reached stable composition mastery
at `4,096` bits versus `8,192` for fresh (`2.0x` fresh-over-inherited), so
the narrow factorized-register composition-transfer claim is promoted. It
still does not establish unrestricted growth, arbitrary program induction, or
general continual learning. The rejected and positive diagnostics are archived in
`session_records/sequence_working_memory_2026-08-02/external_register_interpreter_composition_rejected_v1_2026-08-07/` and
`session_records/sequence_working_memory_2026-08-02/external_register_interpreter_factorized_composition_signal_v1_2026-08-07/`.
The rendered replication is archived in
`session_records/sequence_working_memory_2026-08-02/external_register_rendered_factorized_composition_replicated_signal_v1_2026-08-07/`.

The promoted evidence is in
`session_records/sequence_working_memory_2026-08-02/external_register_rendered_factorized_composition_promoted_v1_2026-08-07/`.

## Three-instruction serial-composition boundary (2026-08-07)

The next pressure test extended the promoted rendered-event register result
to three opaque instructions acquired sequentially without replay. The
primitive-retention gates passed on the acquisition rung: reverse `0.9961`,
complement `0.9648`, and rotate `0.9180` after the third instruction. The
frozen parent digest, exact reload, checksum-corruption rejection, missing-
evidence control, reward-shuffled control, and zero-replay accounting also
passed.

The triple composition itself did not pass. The frozen inherited machine plus
a fresh decoder reached `0.6758`, never crossed the `0.8` stable-prefix gate
in `114,688` unique verifier bits, and its reversed-order control reached
`0.6836`. A matched fresh three-instruction machine reached `1.0000` with a
stable prefix at `16,384` bits. This rejects promotion of depth-three serial
composition while confirming that the failure is downstream of primitive
acquisition and catastrophic-forgetting retention. The short undertrained
rung is archived alongside the acquisition rung in
`session_records/sequence_working_memory_2026-08-02/external_register_three_instruction_rejected_v1_2026-08-07/`.

The implementation bottleneck is now explicit: the external interpreter
combines observation ingestion and repeated program execution in one mutable
register path. A fresh decoder can solve the resulting state when the whole
machine is trained jointly, but a decoder added after three instructions are
learned cannot reliably route it. The next experiment should introduce a
versioned read/execute boundary or an execution snapshot, then re-run the
promoted two-instruction regression before attempting longer programs. This is
not yet arbitrary program induction or general continual learning.

## Promoted read/execute snapshot boundary and depth-three growth (2026-08-07)

The in-place depth-three rejection above isolated the missing state boundary:
the same mutable register held both durable learned observations and repeated
instruction execution results. The production register now exposes
`observe_register()` and `read_execute_register()`. The first persists the
standardized event, opaque feedback, and controller intention. The second
executes the selected opaque instruction chain on a transient snapshot, while
the durable observation state remains free of execution results. The legacy
`step_register()` in-place route remains explicit compatibility; `step()` and
the rendered composition harness use read/execute snapshots.

The original two-instruction regression passed on seeds 69316 and 69317 with
stable inherited composition at `4,096` verifier bits versus `8,192` for fresh
(`2.0x` fresh-over-inherited) on both seeds. The three-instruction
reverse -> complement -> rotate rung then passed on both seeds without replay:
seed 69316 reached stable inherited mastery at `8,192` versus `16,384` fresh
(`2.0x`), and seed 69317 reached `4,096` versus `12,288` fresh (`3.0x`).
Primitive retention, reversed-order composition, reward-shuffled,
missing-evidence, exact-reload, checksum-corruption, frozen-parent, and
zero-replay gates all passed. The archived reports and accounting ledger are
in `session_records/sequence_working_memory_2026-08-02/external_register_read_execute_promoted_v1_2026-08-07/`.

This promotes a bounded read/execute state boundary and three-instruction
compositional growth. It is not arbitrary program induction, unrestricted
memory growth, or general continual learning without catastrophic forgetting.

## Four-instruction nonlinear boundary (2026-08-07)

The next runtime-supplied program was
`reverse -> adjacent_xor -> complement -> prefix_parity`. The canonical
factorized low-rank interpreter retained reverse, complement, and prefix
parity, but adjacent-XOR remained at `0.7734` after the registered acquisition
rung; inherited composition did not beat fresh. A structured factorized FiLM
operator raised minimum primitive retention to `0.8125`, but inherited and
fresh composition tied at `12,288` stable bits. A low-rank-plus-zero-initialized
FiLM hybrid retained all four primitives at `0.9414`, while its serial
composition remained unstable. A deeper 256-update shared blueprint
pretraining phase retained primitives at `0.9336` but collapsed inherited
composition to `0.4805` against fresh `1.0000`.

The controls passed, but no arm passed the positive stable-transfer and
composition gates. This rejects the four-instruction frontier and the tested
nonlinear operator variants. It also gives a useful architectural lesson:
primitive retention and nonlinear expressivity are separable from a stable
serial algebra. The production default remains the promoted factorized
low-rank read/execute path. Evidence and accounting are archived in
`session_records/sequence_working_memory_2026-08-02/external_register_four_instruction_rejected_v1_2026-08-07/`.
A short composition-aware blueprint probe was also rejected: retention was
`0.7813` and inherited composition was `0.7344` versus fresh `0.9844`, so the
curriculum was not scaled.

## Bind-once external operator execution (2026-08-11)

The exported working-memory architecture exposed a general implementation
rule: do context-dependent lookup once, then iterate fixed computation over the
bound result. `ExternalSequenceOperatorMemory.bind(query)` now makes that rule
an explicit versioned ABI. It returns an ephemeral
`BoundExternalSequenceOperatorMemory` containing only the learned route
distribution for the current rollout. The controller and interpreter do not
own the binding, and the external bank remains independently replaceable and
growing.

The shared interpreter accepts the bound handle without a route query or slot
ID. This removes repeated route encoding from multi-step execution while
preserving the raw route-query and fixed-slot forms for diagnostics. A bank
growth event invalidates the active binding and requires rebinding; silently
addressing a changed bank is not permitted. The binding retains gradients to
the route query, so the seam does not turn routing into a detached lookup.

The seed-914 infrastructure audit reduced route calls from 8 to 1 across an
8-step chain, produced exactly equal outputs, preserved route gradients, and
rejected post-growth use until rebinding. It used no verifier bits, optimizer
updates, or replay and therefore makes no learning-capability claim. The
record is
`session_records/external_sequence_operator_bind_once_infrastructure_2026-08-11/`.

This is a reusable execution boundary, not evidence of arbitrary new
computation or general continual learning. The next capability test must use
fresh rendered events and ask whether a bound external file improves depth or
retention under no-replay controls.

The operator bank also has an independent versioned persistence contract:
`payload()` serializes configuration and tensor state with a checksum, while
`from_payload()` validates the schema, dimensions, slot count, state, and
integrity digest before the file can be used. The rendered-event target
harness now consumes this frozen bank rather than merely calibrating it, and
uses paired zero-content corruption plus same-batch reload controls. The first
corrected rung passed exact reload but failed the causal-use gate: bank
corruption changed the two composition targets by only `1.56` and `0.52`
percentage points. This identifies target-side file dependence—not storage or
route binding—as the current bottleneck; the result is archived as a rejected
diagnostic in
`session_records/external_operator_memory_target_bind_rejected_2026-08-11/`.

The follow-up adds a real file-value read path. Each slot now owns independent
`slot_values`, separate from its routing key, and a replaceable
`EpisodicIntentAdapter` conditions the opaque intention on the one-time bound
file token. The adapter is trained during routed external calibration, frozen
before target acquisition, and evaluated with a paired zero-read control. The
v2 file schema remains independently checksummed and accepts legacy v1 files
through an explicit migration path.

The deeper v2 rung passed calibration and exact reload, but the two generated
composition targets still showed `0.00` pp read-ablation drops. The target can
therefore solve the pressure test without using the file token. This rejects
the current read adapter as a learned capability gain while retaining the
useful file ABI and negative diagnostic. The next bottleneck is a target whose
computation is genuinely unavailable except through the external file-read
path; adding more bank width or more slots before that test would only hide the
same bypass.

The v2 report, accounting ledger, and reproduction command are archived in
`session_records/external_operator_memory_read_adapter_rejected_2026-08-11/`.

## Unseen external computation and multi-parent retention (2026-08-06)

The frozen-core transfer harness tested `prefix_parity`, a temporal procedure
outside the earlier forward/reverse/complement transfer set. The shared
controller remained frozen and one generic external recurrent slot learned the
new computation from fresh rendered events, opaque actions, and scalar
outcomes. Seed 69316 reached stable mastery in `12,288` verifier bits versus
`18,432` for a fresh learner (`1.5×` fresh-over-transferred), while seed 69317
reached mastery but tied the fresh learner at `24,576` bits. Parent retention,
frozen-core equality, causal shuffled controls, and zero replay held.

This establishes unseen-procedure acquisition through the generic memory-side
blueprint, not replicated positive transfer. A second-parent rehearsal repair
then preserved the additional parent stream but caused seed-sensitive
interference with the first parent and removed the transfer advantage. The
trainer now supports fresh rehearsal for every mastered parent task, but the
result remains rejected as a general continual-learning solution. The next
implementation target is route-isolated external growth: independent artifact
and decoder paths must prevent a new computation from perturbing old output
residuals. Evidence is in
`session_records/sequence_working_memory_2026-08-02/frozen_core_unseen_prefix_parity_rejected_v1_2026-08-06/`.

## Remediated six-shift external growth to 80 capabilities (2026-08-06)

The repeated-shift pressure test now extends the frozen episodic context and
route boundary from two length-six capabilities through lengths `8, 10, 12,
14, 16, 18`, reaching 80 total capabilities. Across seeds 69316 and 69317,
old routes, new route causality, candidate permutation, reward-shuffled null,
isolated credit, retention reversal/recovery, persistent route/credit reload,
corruption rejection, and zero replay all pass. The weakest shift floors are
`0.8203` and `0.8750`.

The key gain is selective acquisition remediation. A fresh outcome probe
identifies weak rows before admission, and only those external route adapters
receive additional fresh updates. Seed 69317's unremediated family 54 and
seed 69316's late family 68 remain rejected until this targeted evidence is
available; the threshold is not lowered and protected rows are not replayed or
updated. The full bank refuses eviction, then releases and recovers only the
deliberately reversed target.

This was the strongest six-shift bounded growth result, not general continual
learning: the family generator remains finite, each new route still uses an
externally trained blueprint, and arbitrary new computation/open-ended
compression remain unverified. Evidence is in
`session_records/sequence_working_memory_2026-08-02/repeated_shift_growth_6to8to10to12to14to16to18_remediated_v1_2026-08-06/`.

## Remediated seven-shift external growth to 100 capabilities (2026-08-07)

The same isolated external boundary now survives a seventh temporal shift:
length six → eight → ten → twelve → fourteen → sixteen → eighteen → twenty.
Across seeds 69316 and 69317, the bank grows from two base capabilities to
100. Old-route retention, candidate permutation, causal new-route selection,
reward-shuffled null, selective fresh remediation, full-bank
protection/reversal/recovery, persistent route/credit reload, corruption
rejection, and zero replay all pass. The weakest shift floors are `0.8125`
and `0.8594`.

This is a meaningful scale and stability gain for isolated external state, not
a claim of general continual learning. The capability family is still
generated and finite; the controller remains frozen; no positive transfer
against a fresh learner, open-ended compression, or arbitrary new computation
has been established. Evidence is in
`session_records/sequence_working_memory_2026-08-02/repeated_shift_growth_6to8to10to12to14to16to18to20_remediated_v1_2026-08-07/`.

## Generic append-only route isolation (2026-08-06)

The memory-side append boundary is now a reusable
`OpaqueAppendOnlyRouteChain`, rather than a hand-wired two-extension
experiment. It composes any number of independently persisted route
extensions behind the frozen base router. A newly appended route remains
below the established bank until its stage receives scalar failure evidence;
later stages can activate only after all earlier stages have failed. The
controller and base router therefore remain unchanged while external
computation grows.

The generalized audit replicated three protected append boundaries across two
seeds: `rotate4`, `adjacent_xor4`, and `complement_rotate4`, producing six
protected artifacts. Old, append, and combined route accuracy were `1.000` in
both seeds; reward-shuffled append selection was `0.000` for every append;
candidate permutation, causal wrong-artifact behavior, reload, corruption
rejection, frozen digests, and zero replay all passed. This is a stronger
bounded external-growth result and a real route-isolation gain, but it still
does not establish general continual learning, unrestricted memory growth,
arbitrary new computation, or open-ended compression. Evidence is in
`session_records/sequence_working_memory_2026-08-02/multi_append_external_capability_v2_three_step_2026-08-06/`.

## Route-isolated acquisition of an unseen procedure (2026-08-06)

The route-isolated bank then appended `prefix_parity4`, a cumulative temporal
procedure outside the earlier forward/reverse/complement/rotation append set.
Across seeds 69316 and 69317, all seven artifacts remained protected, old and
new route accuracy was `1.000`, reward-shuffled selection was `0.000` for all
four append stages, and selected prefix-parity behavior was `0.8789` and
`0.8359`. Causal wrong-artifact behavior, route and artifact reload,
corruption rejection, frozen parent/base-router digests, and zero replay all
passed.

This is the first replicated evidence that the route-isolated external
blueprint can acquire one computation outside the earlier append family while
preserving prior capabilities. It still does not establish open-ended
program induction: the procedure generator, event cue, recurrent capability
blueprint, and decoder family are fixed by the experiment. The next boundary
is multiple genuinely different unseen procedures and transfer of the
acquisition mechanism itself. Evidence is in
`session_records/sequence_working_memory_2026-08-02/multi_append_external_capability_v3_unseen_prefix_parity_2026-08-06/`.

## Multiple unseen temporal procedures (2026-08-06)

The same route-isolated bank then acquired two distinct unseen temporal
aggregations, `prefix_parity4` and `global_parity4`, after the earlier three
append procedures. Across seeds 69316 and 69317, all eight artifacts stayed
protected, every old/new/combined route was `1.000`, reward-shuffled selection
was `0.000` for every append, and selected behavior for prefix parity was
`0.8789`/`0.8359` while global parity was `1.000`/`1.000`. Causal
wrong-artifact behavior, reload, corruption rejection, frozen parent and
base-router digests, and zero replay all passed.

This strengthens the acquisition claim from one unseen procedure to two
different unseen temporal procedures. It still does not establish open-ended
program induction: the experiment supplies the procedure generator and
operation cues, and every capability uses the same fixed external blueprint
and decoder family. The next bottleneck is therefore mechanism transfer to
procedures that are not predeclared in the append schedule, followed by
capacity growth/compression beyond a fixed-size recurrent artifact blueprint.
Evidence is in
`session_records/sequence_working_memory_2026-08-02/multi_append_external_capability_v4_two_unseen_2026-08-06/`.

## Generated composition artifacts with isolated retention (2026-08-06)

The generated-composition pressure test exposed two useful boundaries. First,
the verifier had to render a generic ordinal cue so noncommutative primitive
orders were learner-identifiable; the previous same-set cue was ambiguous.
Second, a shared external composition stack failed to expand through six fresh
no-replay curriculum phases, while one isolated composition artifact reached
`0.9766` behavior after a stable `16,384` verifier bits.

The next implementation therefore uses the memory boundary directly: each
new generated composition is acquired as an isolated routed external artifact
and admitted to `ExecutableArtifactMemory` only after stable behavior. Fresh
retention outcomes protect the row before transactional growth. An opaque
`OpaqueAddressRouter` then selects among rows from learned event/key
compatibility; candidate-key permutation, reload, corruption, frozen-core,
and replicated reward-shuffled controls are required.

The two-artifact audit promoted this bounded result. Composition behaviors were
`1.0000` and `0.9453`; causal and permuted routing were both `1.0000`; all rows
were protected; reload and corruption checks passed; the parent core was
unchanged; and replay was zero. Three independent reward-shuffled routers,
each tested on two fresh route sets, averaged `0.2580`, so the negative-control
gate passed despite two noisy `0.76–0.78` samples. Evidence is in
`session_records/sequence_working_memory_2026-08-02/generated_composition_artifact_bank_promoted_v1_2026-08-06/`.

This is the first promoted generated-composition artifact-bank result, not
general continual learning. The remaining boundary is a third unseen
composition plus fresh-seed replication, followed by artifacts whose
computation is not drawn from the fixed six-composition grammar and memory
capacity/consolidation beyond a bounded row bank.

## Replicated three-artifact composition growth (2026-08-06)

The append-only generated-composition bank was then extended to composition
ID `2` and rerun with a fresh seed. Across seeds `69316` and `69317`, all three
artifacts mastered (`1.000`/`0.945`/`0.969` and `0.957`/`0.965`/`0.980`), causal
and candidate-permuted route accuracy were `1.000` in both runs, and every
row was protected. Reload, corruption rejection, frozen-core, and zero-replay
gates passed; replicated reward-shuffled controls averaged `0.486` and
`0.394`.

This is now a replicated three-artifact no-replay growth result for the fixed
six-composition grammar. It materially reduces the risk that isolated
external memory growth is merely a one-row or one-seed effect, but it remains
bounded continual artifact growth. The next bottlenecks are a fourth row,
grammar/distribution shift beyond the six predeclared compositions, and
capacity/consolidation when the append-only bank cannot grow indefinitely.
Evidence is in
`session_records/sequence_working_memory_2026-08-02/generated_composition_artifact_bank_three_replicated_promoted_v1_2026-08-06/`.

## Replicated append-only route growth (2026-08-06)

The from-scratch bank router was then replaced in the pressure test by the
generic append-only route chain. One base route was established once; each
new artifact trained only a fresh `OpaqueViewRouteExtension` stage while the
base and earlier stages were frozen. The stage-specific shuffled control was
also corrected to score only the randomized new stage rather than averaging
over unaffected old rows.

Across seeds `69316` and `69317`, three artifact behaviors were
`1.000`/`0.945`/`0.969` and `0.957`/`0.965`/`0.980`; causal route accuracy,
base-key permutation accuracy, cold-start old-route retention, and reload
accuracy were all `1.000` in both runs. Stage-specific shuffled controls were
`0.000` throughout, and protected rows, corruption rejection, frozen-core,
and zero-replay gates passed.

This is the current strongest continual-learning result: replicated
append-only external capability and route growth without updating earlier
route stages. It remains bounded: the six-composition generator and cue
family are predeclared, the artifact blueprint is fixed, and persistent
consolidation beyond append-only capacity is not yet demonstrated. Evidence
is in
`session_records/sequence_working_memory_2026-08-02/generated_composition_artifact_bank_append_only_three_replicated_promoted_v1_2026-08-06/`.

## Append-only acquisition across a grammar shift (2026-08-06)

The same frozen append-only route chain then replaced the fourth family member
with composition ID `6`, a three-primitive `reverse -> complement -> rotate`
program. This tests a longer computation and a changed composition grammar
without changing the controller, base route, or established route stages.

Across seeds `69316` and `69317`, artifact behavior was respectively
`0.9102/0.9688`, `0.8555/0.9414`, `0.9453/0.9609`, and `1.0000/0.9844` for
IDs `0`, `1`, `2`, and `6`. Causal route accuracy, candidate-key permutation,
cold-start old-route retention, and reload were all `1.0000` in both runs.
Every stage-specific reward-shuffled control was `0.0000`; protected-row,
corruption, frozen-core, and zero-replay gates all passed.

This promotes replicated append-only acquisition across a longer-program
grammar shift. It is a meaningful increase in external capability growth, but
not a claim of general continual learning: the program grammar, artifact
blueprint, and append-only memory capacity are still bounded. Evidence is in
`session_records/sequence_working_memory_2026-08-02/generated_composition_artifact_bank_append_only_grammar_shift_replicated_promoted_v1_2026-08-06/`.

## Runtime-supplied grammar mechanism transfer (2026-08-06)

The generated-composition boundary was then generalized so the verifier can
provide a runtime-private grammar rather than selecting only from the static
two- and three-primitive table. The audit supplied four four-primitive
programs: `forward -> reverse -> complement -> rotate`,
`rotate -> complement -> reverse -> forward`,
`complement -> rotate -> forward -> reverse`, and
`reverse -> forward -> rotate -> complement`.

These longer programs were acquired through the same frozen append-only route
chain. Across seeds `69316` and `69317`, artifact behavior was
`0.9219/0.9375`, `0.9141/0.9648`, `0.9609/0.9609`, and `0.9766/0.9492`;
causal route accuracy, candidate-key permutation, cold-start old-route
retention, and reload were `1.0000` in both runs. Every stage-specific
reward-shuffled control was `0.0000`, and protected-row, corruption,
frozen-core, and zero-replay gates passed.

This is replicated mechanism-transfer evidence: the acquisition path now
accepts a withheld runtime grammar and longer programs without changing the
controller interface. It still does not establish arbitrary open-ended
program induction or unrestricted memory growth; the external artifact
blueprint and append-only capacity remain bounded. Evidence is in
`session_records/sequence_working_memory_2026-08-02/generated_composition_artifact_bank_runtime_grammar_replicated_promoted_v1_2026-08-06/`.

## Runtime composition of temporal and aggregation primitives (2026-08-06)

The runtime primitive registry was then extended beyond reversible pointwise
operations to include `adjacent_xor`, `prefix_parity`, and `global_parity`.
Four verifier-private four-primitive programs mixed these operations with
reverse, complement, and rotation:

`reverse -> adjacent_xor -> complement -> prefix_parity`,
`prefix_parity -> global_parity -> rotate -> complement`,
`global_parity -> reverse -> adjacent_xor -> rotate`, and
`complement -> prefix_parity -> reverse -> global_parity`.

Across seeds `69316` and `69317`, artifact behavior was
`0.9336/0.9570`, `1.0000/1.0000`, `1.0000/1.0000`, and `1.0000/1.0000`.
Causal route accuracy, candidate-key permutation, cold-start old-route
retention, and reload were `1.0000` in both runs. Every stage-specific
reward-shuffled control was `0.0000`; protected-row, corruption, frozen-core,
and zero-replay gates passed.

This promotes one compositional external interface for runtime-supplied
temporal and aggregation procedures, strengthening mechanism-transfer evidence
beyond a finite reversible grammar. It remains bounded continual external
growth: the artifact blueprint and append-only capacity are finite, and
arbitrary open-ended program induction is still unverified. Evidence is in
`session_records/sequence_working_memory_2026-08-02/generated_composition_artifact_bank_runtime_nonlocal_replicated_promoted_v1_2026-08-06/`.

## Runtime-generated program mechanism transfer (2026-08-07)

The predeclared append-schedule limitation was then removed from the
generated-composition audit. A deterministic verifier-private generator now
samples distinct four-primitive programs at runtime and rejects functional
duplicates of the fixed grammar before training. The generated tuples are
used only to render ordinary learned event cues and score scalar outcomes;
program tuples, primitive names, and composition IDs never enter the
controller.

With program seed `1739`, the three generated programs were
`reverse -> complement -> rotate -> global_parity`,
`reverse -> global_parity -> reverse -> adjacent_xor`, and
`prefix_parity -> prefix_parity -> rotate -> complement`. Across independent
seeds `69316` and `69317`, all three artifacts were stably mastered and
protected, route and candidate-permutation accuracy were `1.0000`, and the
weakest held-out artifact behavior was `0.8828` and `0.8906`. Reward-shuffled
selection, reload, corruption rejection, frozen-core equality, and zero replay
all passed. The under-budget short and medium controls correctly refused their
first unprotected append.

This promotes runtime-generated mechanism transfer beyond a predeclared
schedule. It remains bounded external growth: the primitive registry,
four-step program depth, artifact blueprint, and append-only capacity remain
finite. Arbitrary open-ended program induction, learned compression,
unrestricted memory growth, and general continual learning remain
unqualified. Evidence and accounting are archived in
`session_records/sequence_working_memory_2026-08-02/generated_composition_artifact_bank_runtime_generated_random_replicated_promoted_v1_2026-08-07/`.

## Runtime-generated eight-step program transfer (2026-08-07)

The runtime-generated procedure boundary was widened from four to eight
ordered primitives. The renderer now places each primitive cue and its opaque
ordinal marker in one of eight event bands, preserving order information
without exposing a composition ID, primitive name, or verifier answer to the
controller.

With runtime seed `2718`, three distinct eight-step programs were acquired
through the same isolated artifact and append-only route path. Across seeds
`69316` and `69317`, all rows were stable and protected, route and candidate
permutation accuracy were `1.0000`, and the weakest artifact behavior was
`0.8711` and `0.9805`. Reward-shuffled selection, reload, corruption
rejection, frozen-core equality, and zero replay all passed. Under-budget
short and medium controls failed artifact mastery, preserving acquisition-depth
evidence rather than weakening the threshold.

This promotes a deeper bounded computational interface, not open-ended
program induction. The primitive registry, eight-step renderer, artifact
blueprint, and append-only capacity remain finite; learned compression,
unrestricted memory growth, and general continual learning remain
unqualified. Evidence and accounting are archived in
`session_records/sequence_working_memory_2026-08-02/generated_composition_artifact_bank_runtime_generated_depth8_replicated_promoted_v1_2026-08-07/`.

## Replicated replay-free positive transfer from external memory (2026-08-06)

The generated-composition harness now compares an inherited external artifact
with a fresh candidate on a genuinely different runtime program. The source
file learned `reverse -> adjacent_xor -> complement -> prefix_parity`; the new
target was `prefix_parity -> global_parity -> rotate -> complement`. Both arms
received fresh target outcomes only. A stable-prefix selector admitted a
candidate only when one arm had a unique verified winner, while the source row
was protected and retained.

Across seeds `69316` and `69317`, the inherited candidate reached stable target
mastery at `6,144` and `4,096` fresh verifier bits, while matched fresh
candidates required `10,240` bits in both runs. The resulting fresh-over-
inherited transfer ratios are `1.667x` and `2.500x`. Source retention floors
were `0.9336` in both runs; the inherited candidate was uniquely selected and
admitted through transactional growth, with frozen controller digests and zero
replay preserved. Source snapshots reloaded with behavior `0.9297` and
`0.9609`, and both corruption controls were rejected.

This is the first replicated positive sample-efficiency transfer result for
the isolated external capability boundary. It does not yet establish broad
general continual learning: the prior/target family, candidate selector, and
fixed artifact blueprint remain bounded, and transfer across arbitrary task
families still requires evidence. Record:
`session_records/sequence_working_memory_2026-08-02/generated_composition_transfer_replicated_promoted_v1_2026-08-06/`.

## Multi-source transfer and finite-capacity logical compaction (2026-08-06)

The next audit increased pressure on the external memory boundary rather than
adding another isolated row. Two independently learned protected files were
compared as initializations for a new runtime program, alongside a fresh
candidate. The source files were
`reverse -> adjacent_xor -> complement -> prefix_parity` and
`global_parity -> reverse -> adjacent_xor -> rotate`; the target was
`prefix_parity -> global_parity -> rotate -> complement`. All target arms
received fresh target outcomes only while the parent controller stayed frozen.

Across seeds `69316` and `69317`, the stable target prefixes were:

| seed | inherited source 0 | inherited source 2 | fresh | selected |
| ---: | ---: | ---: | ---: | --- |
| 69316 | 6,144 bits | 10,240 bits | 10,240 bits | source 0 |
| 69317 | 4,096 bits | 10,240 bits | 10,240 bits | source 0 |

The stable-prefix selector chose source 0 uniquely in both replicas. Before
target admission, the protected two-row bank was rewritten through
`ExternalCapabilityLifecycle.consolidate` into one physical row containing
two opaque namespaced views. The behavior verifier independently reloaded
both views, preserving source behavior at `0.9336/1.0000` and `0.9492/1.0000`.
Both views remained protected, one physical row was saved, and the target was
then admitted by growing capacity from one to two. The grown target reloaded at
`1.0000` in both seeds. Frozen-core digests were unchanged and replayed
examples were zero.

This promotes replicated multi-source bounded external transfer and
behavior-verified logical storage compaction. It is deliberately not called
neural weight compression: the compacted row retains separate executable
views. It also does not establish unrestricted memory growth, arbitrary
program induction, or general continual learning. The short rung correctly
rejected a source view whose fresh retention fell to `0.625`. Evidence and
accounting are in
`session_records/sequence_working_memory_2026-08-02/generated_composition_multi_transfer_replicated_promoted_v1_2026-08-06/`.

## Fresh-outcome neural consolidation of external files (2026-08-06)

The previous multi-source result reduced physical rows while retaining two
separate executable views. The stricter follow-up now tests shared
computation: an inherited student starts from source 0, learns both source
procedures from fresh outcomes, and must replace both protected files with one
routed stack whose aliases resolve to the same artifact digest. A fresh student
receives the same mixed-source budget. When the stable-prefix curves tie, the
inherited state is retained only if a fresh maximin verifier gives it a strict
worst-source behavior margin of `0.02`.

Across seeds `69316` and `69317`, the inherited shared students reached
source behavior `0.9688/1.0000` and `0.9922/1.0000`, while fresh controls
reached `0.7461/1.0000` and `0.8164/1.0000`. The compacted replacement was
`335,456` bytes versus `670,912` bytes for the two source artifacts, a `0.5`
payload ratio. After reload, both aliases resolved to the identical digest and
preserved source behavior at `0.9648/1.0000` and `0.9922/1.0000`; both aliases
remained independently protected. The shared artifact then transferred to the
new target and the grown-bank target reloaded at `1.0000` in both replicas.
The frozen controller was unchanged and replayed examples were zero.

This promotes bounded behavior-verified neural consolidation: one external
network now carries two learned procedures instead of merely co-locating two
networks in one row. The primary stable-prefix selector tied in both replicas,
so the report explicitly records the secondary fresh maximin verifier rather
than misrepresenting the tie as a stable-prefix win. It does not yet establish
arbitrary program induction, unrestricted memory growth, or general continual
learning. The short rung rejected the transaction before source mastery. Full
accounting and reports are in
`session_records/sequence_working_memory_2026-08-02/generated_composition_distilled_consolidation_replicated_promoted_v1_2026-08-06/`.

## Three-source fresh-outcome neural consolidation (2026-08-06)

The shared neural-consolidation boundary now scales from two to three
protected external files. Sources 0, 2, and 3 were learned from fresh
verifier outcomes by one inherited student and compared against a matched
fresh student. The stable-prefix selector was authoritative; inherited weights
were retained when they won that learning-curve comparison, with the strict
fresh maximin behavior margin remaining a tie-only fallback.

Across replicated seeds `69316` and `69317`, the three source rows were
replaced by one shared routed artifact. Its payload was `335,456` bytes versus
`1,006,368` bytes for the three source artifacts, a `0.3333` ratio. All three
aliases resolved to the same artifact digest, remained protected, and passed
fresh reload retention. Source reload behavior was
`0.7578/1.0000/0.8945` and `0.9102/1.0000/1.0000`; the grown target reloaded at
`1.0000` in both replicas. The controller digest was unchanged and replayed
examples were zero.

This is replicated bounded neural consolidation with external capacity growth,
not unrestricted memory growth, arbitrary open-ended program induction, or
general continual learning. The next pressure test is a larger sequential
source set with nonstationary and reversal controls, where retention must hold
without relying on the fixed finite grammar or blueprint. Full reports and
accounting are archived in
`session_records/sequence_working_memory_2026-08-02/generated_composition_distilled_consolidation_three_source_replicated_promoted_v1_2026-08-06/`.

## Four-source replay-free slot-isolated growth (2026-08-06)

The sequential nonstationary pressure test now acquires source procedures
`0`, `2`, `3`, and `4` one at a time. Each new procedure is learned in a
fresh neural slot from fresh verifier outcomes only. The slot is appended into
one physical external artifact row under a new opaque alias; earlier slot
weights, decoders, and retention evidence remain untouched. This makes the
controller/frozen-core boundary useful for continual learning without asking
the core to replay old source streams.

Across replicated seeds `69316` and `69317`, all three `2 -> 1` consolidation
transactions passed behavior verification and reload. The final four aliases
resolved to one physical row, with source behavior
`0.9570/1.0000/0.9531/1.0000` and `0.9805/1.0000/0.9844/1.0000`. The target
was learned from the first retained slot and reached stable mastery at
`2,048` fresh verifier bits in both replicas, versus `14,336` and `8,192` for
matched fresh controls. It reloaded at `1.0000` after capacity growth.

Fresh alias reversal released and recovered only the selected alias while the
shared physical row stayed protected by the other source aliases. The
separate target row released and recovered independently. Checksum corruption
was rejected, the frozen controller digest was unchanged, and replayed
examples were zero. The final payload ratio was `1.0000`, so this is
capacity-safe slot isolation rather than neural compression.

The paired dense shared-weight expansion control was rejected: source 2
reached `1.0000`, but source 0 fell from `0.9531` to `0.6250` when the new
route was trained without old-source replay. This isolates the next real
bottleneck: a new route needs an opaque address/context binding that prevents
interference on old inputs. Evidence and accounting are archived in
`session_records/sequence_working_memory_2026-08-02/generated_composition_sequential_slot_isolated_four_source_replicated_promoted_v1_2026-08-06/`.

## Four-source dense growth with opaque slot binding (2026-08-06)

The rejected dense shared-weight control is now repaired without feeding old
source examples back into the learner. `ExternalCapabilityComposition` has a
versioned optional binding contract, `optional_opaque_external_slot_mask_v1`.
External memory supplies a boolean eligibility mask per alias: an old alias
can use only the slots available at its admission, while the new alias can
use the newly appended slot and earlier slots. This binding is outside the
controller and does not expose source IDs, grammar, task labels, or raw
protocol formats to it.

The four-source sequence `0 -> 2 -> 3 -> 4` was rerun at the matched dense
control budget with seeds `69316` and `69317`. All three append transactions
were adopted; final source behavior after reload was
`0.9570/1.0000/0.9375/1.0000` and `0.9805/1.0000/0.9844/1.0000`. The
inherited target reached stable mastery at `2,048` verifier bits in both
replicas, compared with fresh controls at `14,336` and `8,192`, and reloaded
at `1.0000`. Frozen-core, one-physical-row alias identity, reversal/recovery,
corruption, reload, and zero-replay gates all passed.

This is the first promoted fix for the specific dense-expansion failure: the
old source no longer loses route probability mass merely because a new slot is
trained. It is bounded replay-free dense external growth, not unrestricted
continual learning, neural compression, arbitrary program induction, or
general continual learning. The mask now skips globally ineligible slots, and
a matched full audit preserved every gate and metric while reducing paired
wall time from `961.3s` to `831.4s`. Batch-divergent masks still execute their
active-slot union, so grouped per-mask execution remains a bottleneck.
The action-feedback rollout also now clamps exact-zero logged propensities to
the smallest positive dtype value, fixing a long-run validation failure.
Evidence and accounting are archived in
`session_records/sequence_working_memory_2026-08-02/generated_composition_sequential_route_bound_dense_four_source_replicated_promoted_v1_2026-08-06/`.

The opaque slot binding is now durable memory metadata rather than an
experiment-local map. It survives manifest save/load, growth, compaction,
consolidation, and both promotion paths; a full seed-`69316` audit consumed the
binding from reloaded handles and reproduced every semantic gate and metric.
The audit took `1,244.6s` versus `831.4s` for the in-memory reference, so
batched retention and manifest writes are now the main persistence bottleneck.
Ordered batch retention persistence is now available and reduces a focused
eight-outcome sequence from eight saves to one. The full audit remained
semantically exact, but wall time was noisy, so this is a transaction/write
amortization primitive rather than a promoted end-to-end speedup.
The manifest also carries an atomic SHA-256 sidecar: legacy sidecarless stores
remain readable, but tampered persisted bindings are rejected before execution.
Retention-only outcome updates now persist only the mutable retention ledger;
the structural artifact rows, manifest, and manifest checksum remain untouched.
This narrows the durable write boundary and is covered by a focused invariant
test. A matched four-source audit preserved every semantic gate and metric, but
ran `7.0%` slower end to end, so the change is a correctness/write-scope
improvement rather than a promoted throughput claim. The timing record is
archived in
`session_records/sequence_working_memory_2026-08-02/generated_composition_sequential_route_bound_dense_four_source_replicated_promoted_v1_2026-08-06/report_retention_only_persistence_seed69316.json`.

## Fresh-rebuild neural consolidation of depth-eight procedures (2026-08-07)

The external consolidation boundary now has an explicit capacity-expansion
operation. `expand_routed_stack` copies existing program weights and router
slices exactly, initializes one additional slot with a disabled route, and
keeps the operation outside the frozen controller. The same primitive is used
by sequential growth and distilled consolidation.

The matched depth-eight audit trained two source procedures, compared an
inherited three-slot student with a fresh three-slot student, and enabled the
fresh-rebuild admission path only for this audit. The fresh student won the
stable-prefix selector in both seeds. After independent fresh retention
verification, both source rows were replaced by one shared artifact:

| metric | seed 69316 | seed 69317 |
| --- | ---: | ---: |
| source behavior | `1.0000/1.0000` | `1.0000/1.0000` |
| consolidated reload behavior | `1.0000/1.0000` | `1.0000/1.0000` |
| retention probes | all `1.0000` | all `1.0000` |
| physical rows | `2 -> 1` | `2 -> 1` |
| payload ratio | `0.7392` | `0.7392` |
| controller digest | unchanged | unchanged |
| replayed examples | `0` | `0` |

This promotes replicated bounded fresh-outcome neural consolidation and
behavior-verified external compression for two depth-eight procedures. It is
not inherited positive transfer: both selected students were fresh rebuilds.
Target transfer is unqualified for this promotion, although the seed-69316
diagnostic target reloaded at `0.9961`. Three-source depth-eight controls failed
retention at both 256 and 512 consolidation updates even after slot expansion,
which identifies the next bottleneck as multi-procedure credit assignment and
capacity selection rather than storage plumbing.

The result does not establish arbitrary program induction, unrestricted memory
growth, or general continual learning. Reports, rejected controls, and the
full accounting ledger are archived in
`session_records/sequence_working_memory_2026-08-02/generated_composition_depth8_fresh_rebuild_consolidation_replicated_promoted_v1_2026-08-07/`.

## Staged growth with runtime-generated opaque operators (2026-08-07)

The generated-composition pressure test now includes a verifier-private
operator family independent of the named primitive registry. Each `rule:xx`
token denotes one of the 256 eight-bit functions over the previous, current,
and following binary sequence values. The renderer exposes only a generic
barcode plus ordinal event band; the controller never receives the token,
operator truth table, composition ID, or correct action.

The first run of this audit used an all-slot acquisition mask and therefore
did not prove that a new procedure learned in fresh capacity. The corrected
rerun binds each new source to its newly appended slot while old aliases keep
their prior bindings. Across seeds `69316` and `69317`, both stages adopted;
source reload behavior was `1.0000/1.0000/1.0000` and
`1.0000/0.9844/1.0000`, target reload was `1.0000` in both, and all
reversal/recovery, corruption, frozen-core, exact-reload, and zero-replay
gates passed. This is evidence for strict bounded continual-memory growth
over a newly generated 256-member local-rule family, not arbitrary Turing
complete computation, unrestricted memory growth, or general continual
learning. The authoritative reports and accounting ledger are archived in
`session_records/sequence_working_memory_2026-08-02/generated_composition_opaque_rule_sequential_slot_growth_three_source_strict_isolated_replicated_promoted_v1_2026-08-07/`; the permissive predecessor remains audit history.

## Replay-free staged growth for three depth-eight procedures (2026-08-07)

The historical staged external-growth path scaled the procedure family to
three runtime-generated eight-step procedures. It froze old slot parameters
and route slices, but its all-slot masks allowed a new source to use old
outputs; it therefore does not prove isolated fresh-capacity acquisition. The
memory transaction still adopted a stage only after fresh probes verified
both the new alias and every previously retained alias.

Across seeds `69316` and `69317`, both sequential stages were adopted. The
final source aliases survived reload at `1.0000/1.0000/0.8789` and
`1.0000/1.0000/0.8047`; the target was learned from the retained artifact and
reloaded at `1.0000` in both seeds. Reversal/recovery, corruption, exact
reload, frozen-core, and zero-replay gates all passed.

This is a historical bounded retention result, not the strict isolated-slot
claim: new mutable state was added without updating old capability weights,
but old outputs were not excluded during acquisition. It is not yet general
continual learning. The primitive registry, eight-step renderer, slot
blueprint, and tested horizon remain finite. Reports and accounting are in
`session_records/sequence_working_memory_2026-08-02/generated_composition_depth8_sequential_slot_growth_three_source_replicated_promoted_v1_2026-08-07/`.

## Ten-procedure strict isolated growth over opaque runtime rules (2026-08-07)

The strict external slot boundary now acquires nine verifier-private,
runtime-generated eight-step opaque procedures sequentially, then admits a
tenth target into a separate grown row. Across seeds `69316` and `69317`, all
nine fresh slot transactions adopted; every retained alias survived fresh
reload verification; target reload was `1.0000`; reversal/recovery,
corruption rejection, exact bank reload, frozen-core equality, and zero replay
all passed. The weakest final source behaviors were `0.9141` and `0.8320`.

This is a real scale gain for frozen-core external memory: the deployed
controller still receives only learned events, opaque feedback, and scalar
outcomes, while slot identity and rule semantics remain in external memory.
It is not yet general continual learning. The representation is still one
fresh neural slot per procedure with linear payload growth; inherited target
transfer did not beat fresh learning, and shared computation, learned
compression, arbitrary open-ended program induction, and unrestricted growth
remain unverified. Evidence is in
`session_records/sequence_working_memory_2026-08-02/generated_composition_opaque_rule_sequential_slot_growth_ten_source_runtime_generated_replicated_promoted_v1_2026-08-07/`.

## Shared-base residual growth boundary (2026-08-07)

`ExternalCapabilitySharedResidualBank` provides a replaceable memory-side
contract with one shared `EpisodicContextEncoder` and independently frozen
residual intention adapters. `step_slot` executes one opaque binding using
only that slot's external recurrent state; `step` preserves all slot states
in a versioned pipeline state. The shared base can be frozen explicitly before
adding a slot, and protected residuals can be frozen independently.

The replicated registry audit promotes two related procedures with a `0.5556`
payload ratio versus independent full programs, exact reload, old-slot
retention, frozen-core equality, and zero replay. Matched controls reject two
genuinely opaque random procedures and a third heterogeneous registry
procedure. Therefore the contract verifies reusable shared computation, but
also identifies its current limit: an adapter-only residual cannot supply
arbitrary new sequential computation once the shared basis is frozen. The next
growth mechanism must add append-only compute capacity or verified compressed
behavioral summaries without mutating protected capabilities. Evidence is in
`session_records/sequence_working_memory_2026-08-02/generated_composition_shared_residual_bank_registry_replicated_boundary_v1_2026-08-07/`.

## Append-only residual compute for opaque acquisition (2026-08-07)

`ExternalCapabilityResidualComputeBank` extends the shared-base boundary with
a compact recurrent context encoder per appended slot. The shared encoder and
protected slots remain immutable, while the new slot gets enough local
sequential computation to learn a procedure that is not represented by the
frozen basis. Its state is still an external, versioned capability state and
the controller sees only the standardized event/action/outcome/intention
boundary.

At local hidden/width `32/16`, a two-seed opaque-rule audit reaches reloaded
behavior `0.8906` and `0.9023` for the new procedure while retaining the old
procedure at `1.0000`. Exact reload, deliberate corruption recovery,
shared-base/old-slot immutability, frozen-core, and zero-replay gates pass.
This is a capability gain, not general continual learning: unique procedures
still consume local compute slots and the two-slot payload remains `0.8221` of
independent full programs. The next bottleneck is verified reuse or
compression of local compute across later procedures. Evidence is in
`session_records/sequence_working_memory_2026-08-02/generated_composition_shared_residual_compute_opaque_replicated_promoted_v1_2026-08-07/`.

The residual-compute bank now survives a third opaque procedure. Across seeds
`69316` and `69317`, all three aliases reload at
`1.0000/0.8906/1.0000` and `1.0000/0.9023/1.0000`; every prior slot remains
protected, the shared base remains unchanged, and corruption recovery,
frozen-core, exact-reload, and zero-replay gates pass. The three-slot payload
is `0.6740` of independent full programs. This advances the bounded external
compute boundary but leaves open-ended growth and learned cross-slot
compression unverified. Evidence is in
`session_records/sequence_working_memory_2026-08-02/generated_composition_shared_residual_compute_opaque_three_replicated_promoted_v1_2026-08-07/`.

The residual-compute bank now survives a fourth opaque procedure. Across two
seeds, all four reload at `1.0000/0.8906/1.0000/0.9766` and
`1.0000/0.9023/1.0000/0.9453`; every earlier slot remains protected and the
external-memory checksum detects and restores deliberate corruption. The
four-slot payload is `0.5999` of independent full programs. This is bounded
external compute growth; reuse/compression of local compute and unrestricted
continual learning remain open. Evidence is in
`session_records/sequence_working_memory_2026-08-02/generated_composition_shared_residual_compute_opaque_four_replicated_promoted_v1_2026-08-07/`.

## Reusable physical compute and logical bindings (2026-08-07)

`ExternalCapabilityReusableComputeLibrary` separates a physical compact
recurrent compute module from logical capability bindings. Each binding owns
its intention adapter and external recurrent state, while an opaque
memory-side table points it at a physical compute slot. A new binding can
reuse a module without copying or updating its weights; a new physical module
is added only when fresh verification rejects reuse.

The replicated registry audit binds two related procedures to one physical
module, retains both after reload, and passes independent-state, checksum,
frozen-core, and zero-replay gates. The matched opaque reuse control rejects
at `0.6367`. This is verified selective compute reuse, not arbitrary program
compression: incompatible procedures still require new local compute. Evidence
is in
`session_records/sequence_working_memory_2026-08-02/generated_composition_reusable_compute_registry_replicated_promoted_boundary_v1_2026-08-07/`.

## Fresh-verified reuse-first, grow-on-failure admission (2026-08-07)

`select_reusable_compute_slot` is the versioned memory-side admission policy.
It scores opaque physical candidates by their worst fresh retention probe and
reuses only candidates whose every probe clears the mastery floor. If none
passes, the binding is discarded and a new physical compute module is grown.

The replicated audit reuses one module for both registry procedures. For the
opaque pair, one seed rejects reuse and grows a second module, while the other
accepts reuse because its fresh probes pass. All four outcomes retain after
reload with frozen-base, old-binding, checksum, frozen-core, and zero-replay
gates. This is adaptive bounded compute admission, not a semantic classifier
or general continual learning. Evidence is in
`session_records/sequence_working_memory_2026-08-02/generated_composition_reuse_first_grow_policy_replicated_boundary_v1_2026-08-07/`.

## Multi-candidate fresh-verified compute reuse (2026-08-07)

The reusable library now supports content-addressed candidate admission. A
new binding is trained in an isolated trial against every physical compute
slot; `select_reusable_compute_slot` chooses the best worst-case fresh score,
or all trials are discarded and a new compute slot is grown. The logical
binding table remains external and versioned.

The replicated three-opaque-procedure audit passes with a `2:3` physical to
logical ratio at seed `69316` and `1:3` at seed `69317`. All prior bindings
retain after reload, checksum restoration and frozen-core gates pass, and
replay is zero. This is adaptive bounded compute reuse, not open-ended
compression or general continual learning. Evidence is in
`session_records/sequence_working_memory_2026-08-02/generated_composition_candidate_reuse_opaque_three_replicated_promoted_v1_2026-08-07/`.

## Generalizing learned compute-candidate screening (2026-08-07)

`LearnedComputeCandidateScreen` is the parametric counterpart to the
exact-context/global-prior screen. It uses a factorized opaque query/key
scorer, receives learned event tensors and opaque external compute keys, and
learns from attempted scalar outcomes using pairwise candidate ranking. It is
disabled at cold start and must be explicitly enabled only after evidence;
it remains an ordering aid and cannot authorize reuse without the fresh
verifier admission policy.

In a two-seed six-candidate pressure test, four known candidates are trained
from scalar outcomes and two candidates remain outcome-unseen. Fresh novel
contexts over the known candidates route at `1.0000` for both seeds versus
`0.2500` cold-start; candidate permutation, exact reload, frozen-core, and
reward-shuffled null gates pass. The outcome-unseen candidates route at
`0.0000`, which is retained as the explicit cold-start control rather than
overclaimed as generalization. This promotes learned query generalization for
known external candidates, not acquisition of arbitrary unseen computation or
general continual learning. Evidence is in
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_screen_opaque_novel_context_replicated_promoted_v1_2026-08-07/`.

A matched registry-family control fails the novel-context and candidate
permutation gates at `0.2500`; representation transfer across family
distributions remains open.

## Shared-screen unseen-candidate calibration rejected (2026-08-07)

Applying fresh outcomes for two previously unseen candidates directly to the
shared learned screen acquires them at `1.0000` across both seeds, but known
candidate routing falls from `1.0000` to `0.2083/0.2500` and candidate
permutation fails. The frozen controller is unaffected, but the external
screen itself catastrophically forgets. This rejects shared screen mutation
as the acquisition mechanism; the next boundary must append isolated screen
state and gate its activation. Evidence is in
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_screen_calibration_rejected_v1_2026-08-07/`.

## Append-only learned screen growth promoted (2026-08-07)

The safe replacement was tested on the same six-candidate opaque-rule
pressure test. The mastered screen is copied into an independently versioned
base and frozen; two new candidates receive an isolated learned extension.
Before a scalar verifier failure, the extension is forced below the base and
the unseen-candidate top-1 rate is `0.0000`. After that failure, 64 fresh
outcome updates raise unseen-candidate top-1 to `1.0000` on both seeds while
known-candidate novel-context routing remains `1.0000`. Base and extension
candidate permutations, exact reload, frozen-core, reward-shuffled null, and
zero-replay controls pass. The extension is still an order-only aid and fresh
verifier admission remains authoritative. This promotes safe append-only
screen growth, not unrestricted memory growth or general continual learning.
Evidence and accounting are in
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_screen_append_only_promoted_v1_2026-08-07/`.

## Multi-stage append-only learned screen growth promoted (2026-08-07)

The learned append boundary now scales to two sequential isolated stages. In
a ten-candidate pressure test, six mastered candidates remain in the frozen
base and four outcome-unseen candidates are split across two two-candidate
extensions. Each later stage can activate only after the base and all earlier
stages have produced scalar verifier failure. Across seeds `69316` and
`69317`, unseen routing is `0.0000` before activation and `1.0000` after the
cumulative failure schedule; known routing is `1.0000`, both stage-local
candidate permutations are exact, and reload, frozen-core, reward-shuffled,
and zero-replay gates pass. This promotes replicated multi-stage bounded
growth, not unbounded memory, arbitrary new computation, or general
continual learning. Evidence and accounting are in
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_screen_append_only_two_stage_promoted_v1_2026-08-07/`.

A pairwise-only singleton-stage control is rejected: two stages with one
candidate each remain at `0.5000` post-failure routing even after 256 updates
per stage, because pairwise ranking receives no informative within-stage
comparison. The later attempted-outcome calibration objective repairs this
boundary; the control is retained in
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_screen_append_only_singleton_stage_rejected_v1_2026-08-07/`.

## Cardinality-independent append calibration promoted (2026-08-07)

The learned append boundary now uses pairwise ranking when a stage has
multiple candidates and attempted-outcome calibration when it has one. In a
mixed ten-candidate audit, the first extension has one candidate and the
second has two; the unary stage receives fresh positive and negative scalar
verifier attempts, with no unattempted-candidate labels. Across seeds `69316`
and `69317`, pre-activation unseen routing is `0.0000`, post-failure routing
is `1.0000`, known routing is `1.0000`, and base/stage-local permutation,
reload, frozen-core, reward-shuffled, and zero-replay gates pass. This
promotes a cardinality-independent bounded append mechanism, not arbitrary
program induction or general continual learning. Evidence and accounting are
in
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_screen_cardinality_independent_mixed_promoted_v1_2026-08-07/`.

The same mixed `[1, 2]` boundary then passes at 128 fresh calibration updates
per stage across both seeds, halving append calibration optimizer updates
while preserving every gate. A blind copy of the base address weights into
new extensions is rejected at the same budget (`0.3333/0.6667` acquisition),
so fresh extension state remains canonical until a selective prior earns
causal evidence. Evidence is in
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_screen_cardinality_independent_mixed_128_promoted_v1_2026-08-07/`
and
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_screen_inherited_prior_rejected_v1_2026-08-07/`.

Selective query-side transfer then passes the same mixed `[1, 2]` audit at
64 updates per stage across both seeds: only query projections are copied,
while candidate-key and matching state remain fresh. The matched fresh
control fails seed `69317` at `0.6667` and needed 128 updates per stage for a
two-seed promotion. This promotes a 50% acquisition-cost reduction without
retaining blind inherited weights. Evidence is in
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_screen_selective_query_prior_promoted_v1_2026-08-07/`.

The same selective-prior boundary was then extended to three sequential
singleton stages. At 64 calibration updates per stage, both seeds fail the
strict unseen-acquisition gate (`0.3333/0.6667`); at 128, one seed passes and
one fails (`0.8125/0.7396`). At 256, both seeds pass all gates with `1.0000`
unseen routing, known-context retention, stage-local permutation, reload,
frozen-core, reward-shuffled, and zero-replay controls. The matched fresh
three-stage control also passes both seeds at 256, so this promotes replicated
three-stage bounded append growth but does not extend the selective prior's
positive efficiency claim to three stages. Stage-wise calibration/sample
efficiency is now the bottleneck as sequential depth increases. Evidence and
the rejected lower-budget controls are in
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_screen_append_only_three_stage_boundary_v1_2026-08-07/`.

The inherited query path is now tunable through a copy-on-write prior
strength. Half-strength transfer repairs the one-seed three-stage failure at
128 updates (`0.8542/0.9063`), while quarter strength remains unstable
(`1.0000/0.6667`). However, half strength fails the three-stage 64-update
boundary and also loses one seed on the earlier mixed two-stage 64-update
boundary. No universal strength is promoted: the result establishes a safe
escape mechanism for harmful inherited basins and leaves prior selection as
an external evidence-driven policy problem. Evidence is in
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_screen_prior_strength_boundary_v1_2026-08-07/`.

The same append boundary then scales the opaque bank from ten to fourteen
candidates: eight mastered base rows and six outcome-unseen rows split across
three isolated two-candidate stages. At 32 calibration updates per stage,
both fresh and full query-prior initialization pass both seeds with unseen
routing of `1.0000/0.8333` and `1.0000/0.9063`, respectively. Retention,
stage-local permutation, reload, frozen-core, reward-shuffled, and zero-replay
gates pass. Because fresh initialization also passes, this promotes bounded
cardinality scaling rather than prior efficiency or unrestricted continual
learning. Evidence is in
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_screen_append_only_bank14_three_stage_32_promoted_v1_2026-08-07/`.

Sequential capacity then reaches five isolated two-candidate stages in a
twenty-candidate bank: ten mastered base rows plus ten outcome-unseen rows.
At 32 updates per stage, both fresh and full query-prior controls pass both
seeds with unseen routing `1.0000/0.8958`; all retention, permutation,
reload, frozen-core, reward-shuffled, and zero-replay gates pass. This
promotes repeated bounded append growth, but external state still grows
linearly and no prior-efficiency, learned-consolidation, unrestricted-growth,
or general-continual-learning claim follows. Evidence is in
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_screen_append_only_bank20_five_stage_32_promoted_v1_2026-08-07/`.

The append screen now exposes `consolidate_verified` as a transactional
memory-side boundary. A caller supplies independently trained replacement
state for consecutive extensions and a fresh behavior verifier; the source
screen is immutable, logical candidate count is preserved, and only an
accepted candidate compacts physical extension modules. This establishes the
safe compaction contract, not behavioral consolidation itself: a learned
replacement still needs a fresh-outcome audit before adoption.

The first behavior-level consolidation audit is rejected. A four-candidate
replacement for two two-candidate stages was trained from fresh scalar
outcomes plus source-route distillation, but it failed strict repeated
per-candidate retention on both seeds. One seed also showed that the source
bank itself had two logical candidates at `0.0000` per-target retention, so
compaction cannot repair upstream mastery. A naïve copied-stage replacement
was rejected on both seeds. This keeps the compaction contract while rejecting
the current learned replacement path; evidence is in
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_screen_consolidation_rejected_v1_2026-08-07/`.

A more expressive pairwise candidate router is also rejected at the same
twenty-candidate/five-stage/32-update boundary: unseen routing is
`0.9063/0.7083`, below the factorized baseline's `1.0000/0.8958`, and one seed
fails promotion. The extra branch is removed from the canonical screen; the
remaining bottleneck is candidate-signature separation and per-candidate
mastery, not scorer expressiveness. Evidence is in
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_screen_pairwise_router_rejected_v1_2026-08-07/`.

The factorized boundary was then rerun with strict per-candidate promotion.
Seed `69316` clears all ten unseen targets, but seed `69317` has aggregate
unseen accuracy `0.8958` with one target at `0.0000`, and known-target holes at
`0.7000` and `0.0000`. Candidate-key diagnostics expose the representation
failure directly: nearest-neighbor cosine reaches `0.9956/0.9982` and
effective rank is only `4.47/3.59` for twenty keys. The replicated boundary is
therefore not promoted until upstream signatures and source mastery are fixed;
aggregate routing is no longer accepted as sufficient evidence. See
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_screen_per_candidate_mastery_audit_v1_2026-08-07/`.

A spatial-binding frontend control improves key separation without improving
acquisition: effective rank rises to `7.01/6.16` and worst nearest-neighbor
cosine falls to `0.9881/0.9860`, but unseen routing falls to `0.8958/0.8021`
and both seeds fail strict per-target mastery. The blueprint is retained as a
diagnostic lead, while the promoted path remains unchanged until query/key
alignment and downstream generalization improve together. Evidence is in
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_screen_spatial_binding_frontend_rejected_v1_2026-08-07/`.

The matched full-prior control repairs unseen routing at this rung
(`1.0000/1.0000`) but leaves the seed-`69317` source bank at `0.8542` known
routing with per-target holes at `0.7000` and `0.0000`. It is therefore not a
replicated promotion: append initialization is no longer the immediate
blocker; source-screen mastery and query/key alignment are. Evidence is in
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_screen_full_prior_strict_rejected_v1_2026-08-07/`.

Doubling only the source-screen budget to 1024 updates and retaining the full
append prior now passes the strict bank-20/five-stage boundary across both
seeds: known and unseen routing are `1.0000/1.0000` with every audited target
above the mastery floor. The matched fresh-extension control passes one seed
but fails the other with one unseen target at `0.0000`, so full prior transfer
is retained as a bounded robustness mechanism. This remains finite external
growth, not unrestricted continual learning; effective key rank on the hard
seed is still only `3.59`. Evidence is in
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_screen_source_mastery_full_prior_1024_promoted_v1_2026-08-07/`.

Giving the spatial-binding control twice the local append calibration budget
(64 updates per stage) leaves the hard seed unchanged at `0.8125`, with two
unseen targets at `0.0000`; the easy seed remains `1.0000`. Extra local
calibration is therefore rejected as a remedy, further localizing the failure
to static query/key alignment. Evidence is in
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_screen_spatial_binding_calibration_rejected_v1_2026-08-07/`.

The memory boundary now exposes `LearnedOpaqueCandidateKeyMemory`: an
appendable, independently freezeable address store whose parameters can be
trained from attempted scalar outcomes outside the controller. The first
joint-update control is rejected because it destabilizes the mastered base and
falls to `0.5/0.5` unseen routing. The safer extension-only update preserves
the base and passes the strict floor, but slightly regresses the easy seed
(`0.9792` versus `1.0000`) and is not accepted as an improvement. The API is
retained as a contract; its behavioral update rule remains open. Evidence is
in
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_screen_naive_key_adaptation_rejected_v1_2026-08-07/`
and
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_screen_extension_key_memory_rejected_v1_2026-08-07/`.

The next capacity rung—26 candidates with 12 unseen rows across six append
stages—still acquires every unseen candidate at `1.0000/1.0000`, but source
known routing falls to `0.9271/0.7500` and strict per-target mastery fails on
both seeds. The 26-candidate boundary is therefore source-screen capacity and
interference limited, not append-limited. Evidence is in
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_screen_bank26_six_stage_1024_rejected_v1_2026-08-07/`.

Doubling the bank-26 source budget again to 2048 updates still leaves seed
`69316` with known-target holes at `0.7143`, `0.2857`, and `0.5714`, while
unseen acquisition remains `1.0000/1.0000`. Seed `69317` passes, so this is
not a replicated promotion. The result rejects budget scaling as the next
remedy and points back to representation/interference. Evidence is in
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_screen_bank26_six_stage_2048_rejected_v1_2026-08-07/`.

A frozen permutation-invariant affine signature normalizer was then tested as
an upstream representation repair. It restores bank-26 source mastery on both
seeds, but unseen acquisition falls to `1.0000/0.8854` on the hard seed. A
dual raw-plus-normalized signature view is worse at `0.9167/0.7396` unseen
accuracy and also loses known-target mastery. Neither global representation
change is promoted. The evidence points to page-local representation
selection with verifier-gated routing, rather than concatenating or replacing
one address space for every capability. Reports are in
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_screen_affine_normalizer_rejected_v1_2026-08-07/`
and
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_screen_dual_signature_rejected_v1_2026-08-07/`.

The page-local memory ABI resolves that boundary without changing the
controller: the source page uses the frozen affine-normalized view, while six
append pages use raw identity views and representation-matched copy-on-write
priors. A local rank-margin activation prevents incompatible page score scales
from suppressing an extension after its cumulative verifier failure gate opens.
At the matched bank-26 pressure test, both seeds pass strict known and unseen
per-candidate mastery (`1.0000/1.0000`), candidate permutation, reload, null,
frozen-base, and frozen-core controls. The promoted budget is 1,024 normalized
source updates, 512 raw-prior updates, and 32 fresh calibration updates per
append stage, with zero replayed examples. This is a structural bounded-growth
promotion, not learned representation selection or general continual learning.
The 512/512 source-budget split retains perfect unseen acquisition but fails
hard-seed known retention at `0.8854` with three target holes, so the normalized
source page still needs the full source budget. Reports and accounting are in
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_screen_page_local_representation_promoted_v1_2026-08-07/`
and
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_screen_page_local_representation_budget_split_rejected_v1_2026-08-07/`.

Pushing the same page-local budget to 46 candidates—20 protected source rows
and 26 unseen rows across 13 append stages—replicates a new source boundary:
both seeds retain `1.0000/1.0000` unseen acquisition but known routing is
`0.9271` with a `0.4000` per-target floor. Normalized key effective rank rises
to `11.90/12.87`, so the next remedy is source-router capacity/interference,
not another global signature transform. Evidence is in
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_screen_page_local_46_source_boundary_rejected_v1_2026-08-07/`.

Increasing the page-local source scorer latent width from 32 to 64 does not
repair this boundary: known floors are `0.0000/0.4000` across the two seeds,
while unseen acquisition remains perfect. The extra latent coordinates are
therefore not the next lever; router hidden capacity or isolation of mutable
source competition is. Evidence is in
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_screen_page_local_46_latent64_rejected_v1_2026-08-07/`.

Increasing router hidden width from 64 to 128 also worsens the 46-candidate
source route (`0.8542/0.8021` aggregate known accuracy, `0.0000/0.0000`
per-target floors) while unseen acquisition remains perfect. Capacity scaling
alone is rejected; the next intervention isolates source competition into
independent pages behind verifier-gated activation. Evidence is in
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_screen_page_local_46_hidden128_rejected_v1_2026-08-07/`.

Source competition is now isolated into physical memory pages. Splitting the
20 protected source rows into two independently trained normalized pages of
ten, with the second page opened by scalar failure, restores strict 46-
candidate retention: both seeds pass all 20 known and 26 unseen targets at
`1.0000/1.0000`, including permutation, null, reload, source-page immutability,
and zero-replay controls. The run uses 2,496 optimizer updates and 477,696
verifier bits, versus 3,008 updates and 1,423,872 bits for the unsharded
page-local control. This promotes bounded source-competition isolation and a
sample-efficiency gain, not learned page retrieval or general continual
learning. Evidence is in
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_screen_page_local_source_sharded_46_promoted_v1_2026-08-07/`.

The same source-page mechanism scales to 64 candidates: three normalized
source pages protect 30 rows and 17 raw append pages acquire 34 unseen rows.
Both seeds pass all strict known/unseen per-target, permutation, null, reload,
page-immutability, and zero-replay gates at `1.0000/1.0000`. Accounting is
3,136 optimizer updates and 652,800 verifier bits. This extends bounded source
competition isolation; page order is still physical and learned page retrieval
remains open. Evidence is in
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_screen_page_local_source_sharded_64_promoted_v1_2026-08-07/`.

The next boundary is now learned page addressing. A router trained only from
the scalar verifier outcomes of attempted local pages retrieves three
independently trained normalized source pages at 64 candidates. Both seeds
reach strict `1.0000` candidate and page retrieval, every target/page is
mastered, page order can be permuted without loss, reward-shuffled outcomes
produce the null, reload is exact, the controller is unchanged, and replay is
zero. Each run uses 2,592 optimizer updates, 221,952 verifier bits, and
221,568 logical lifetimes. This promotes bounded learned external page
addressing; arbitrary memory growth, learned representation selection, unseen
append integration, and general continual learning remain open. Evidence is
in
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_page_router_64_promoted_v1_2026-08-07/`.

The direct append-only control rejects freezing that router and merely adding
new page keys. With 34 new candidates in 17 external pages, accuracy falls to
`0.3958` and `0.1250` across the two seeds, with zero mastery on most append
pages, despite frozen source state, unchanged controller, and zero replay. The
failure shows that external growth needs a learned address-update mechanism;
the rejected evidence is in
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_page_router_append_frozen_rejected_v1_2026-08-07/`.

The token-preserving append-router overlay closes that bounded growth
boundary. The source router and source pages remain frozen; a separate
append-router retains every normalized opaque candidate token instead of
collapsing each page to a mean. Scalar verifier failure gates fallback from
the source router to the append router. With 34 new candidates in 17 pages,
both matched seeds pass strict `1.0000` candidate/page and per-target/per-page
mastery, full page permutation, reward-shuffled null, frozen-source, unchanged
controller, and zero-replay gates. Each run uses 10,816 optimizer updates,
1,887,744 verifier bits, and 1,887,360 logical lifetimes. This promotes
bounded no-replay external page addressing, not arbitrary memory growth,
compression, learned representation selection, or general continual learning.
Evidence is in
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_page_router_append_token_overlay_64_promoted_v1_2026-08-07/`.

The same overlay now passes with a reduced append budget: 3,072 updates for
each append router rather than 4,096. Both seeds still pass strict
`1.0000` candidate/page and per-target/per-page mastery plus permutation,
shuffled-null, frozen-source, unchanged-controller, verifier-fallback, and
zero-replay gates. The new accounting is 8,768 optimizer updates, 1,469,952
verifier bits, and 1,469,568 logical lifetimes per seed. This supersedes the
prior cost boundary but remains bounded external page addressing. Evidence is
in
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_page_router_append_token_overlay_64_promoted_v2_2026-08-07/`.

Repeated growth now passes a two-generation audit. Thirty source candidates
remain in three normalized pages; two independent append generations add 18
candidates each in nine raw pages, for 66 candidates and 21 pages total. Each
generation has its own token-preserving router trained only on its own scalar
verifier outcomes, and inference cascades source → generation 1 → generation 2
on failure. Both seeds pass strict `1.0000` candidate/page and per-target/
per-page mastery, full permutation, generation-local shuffled nulls, frozen
source state, unchanged controller, no unresolved rows, and zero replay. Each
run uses 14,944 optimizer updates, 1,544,448 verifier bits, and 1,544,064
logical lifetimes. This promotes bounded repeated external growth, not
unrestricted memory growth, consolidation/compression, or general continual
learning. Evidence is in
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_page_router_two_generations_66_promoted_v1_2026-08-07/`.

The same two-generation audit now fits the frozen signature normalizer only
on the 30 original source keys, keeping future append keys out of the
representation boundary. At a matched 4,096 updates per generation router,
both seeds pass all strict gates, including generation-local shuffled nulls
and zero replay. Each run uses 19,040 optimizer updates, 1,986,816 unique
verifier bits, and 1,986,432 unique logical lifetimes. The 3,072-update
source-only attempt is retained as a rejected budget boundary because one
seed missed one target. This promotes a source-only representation contract
for bounded repeated growth; unrestricted memory growth, consolidation/
compression, arbitrary new computation, and general continual learning
remain open. Evidence is in
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_page_router_two_generations_66_source_normalizer_promoted_v1_2026-08-08/`.

The source-only contract then survives a third independent append generation:
30 source candidates plus three 18-candidate generations reach 84 candidates
across 30 pages. Both seeds pass the same strict per-target/per-page,
permutation, generation-local shuffled-null, source-immutability,
controller-invariance, reload, and zero-replay gates. Each run uses 21,376
optimizer updates, 2,214,912 unique verifier bits, and 2,214,528 unique
logical lifetimes; mean fresh attempts rise to `2.2857` with a `0.6429`
fallback rate. This confirms bounded repeated growth while exposing the next
cost boundary: routing state and verifier work remain linear. Consolidation,
compression, unrestricted growth, arbitrary new computation, and general
continual learning remain open. Evidence is in
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_page_router_three_generations_84_source_normalizer_promoted_v1_2026-08-08/`.

A direct consolidation attempt is rejected. Replacing two generation-specific
routers with one flat router over all 18 append pages reaches only `0.50/0.50`
generation accuracy at latent 32/hidden 64; increasing it to latent 64/hidden
128 reaches `0.50/0.5682`, with permutation only `0.50/0.5341`. Both fail
per-target and per-page retention despite near-zero training loss, while the
original cascade remains perfect and the reward-shuffled null remains valid.
This identifies shared-page interference rather than a simple width deficit.
The rejected evidence is in
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_page_router_flat_consolidation_rejected_v1_2026-08-08/`.
The next consolidation design is factorized: a shared local-page router plus
a small generation selector, preserving generation separation without one
full router per generation.

The factorized design is also rejected at the 84-candidate/three-generation
boundary. One shared local-page router with identity-initialized generation
adapters reaches only `0.679/0.679/0.643` on the three generations; its
verifier-gated cascade reaches `0.6667` overall with unresolved rows and
`0.6667` page-permutation accuracy. A full-token generation selector with
4,096 updates does not repair the result. The reward-shuffled cascade remains
a valid null, so this is retention failure rather than verifier leakage.
Page-router consolidation is therefore not promoted; the next compression
intervention should target the verified artifact-level reusable-compute
library instead. Evidence is in
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_page_router_factorized_consolidation_rejected_v1_2026-08-08/`.

The reusable compute ABI now also permits multiple opaque bindings to point at
one physical intention adapter, while each binding retains isolated recurrent
state. This is a memory-side capability for verifier-approved compatible
bindings, not a semantic classifier or an unconditional merge; incompatible
bindings can still allocate a fresh adapter or compute slot. Behavioral
verification must precede any sharing decision.

The admission policy is explicit in `select_reusable_binding`: it scores each
opaque `(compute_slot, adapter_slot)` candidate using the minimum fresh-probe
outcome and reuses the best candidate only when every probe clears the verifier
floor; otherwise it returns `grow`. This keeps adapter sharing memory-side and
verifier-gated, without claiming behavioral compression before the compatible
sharing audit is promoted.

The first two-seed adapter-sharing audit is retained as negative evidence in
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_adapter_sharing_rejected_v1_2026-08-08/`.
It found a verifier-passing shared adapter for one later capability in both
runs, but the incompatible middle capability failed strict mastery in both.
Adapter sharing is therefore an available ABI mechanism, not a promoted
continual-growth result; the next audit must improve incompatible-capability
growth and include a matched source-order permutation control.

The follow-up growth-choice protocol is promoted for the bounded canonical
three-capability sequence across seeds `69316` and `69317`. After a failed
adapter-sharing probe it verifier-scores both fresh-adapter growth and fresh
compute-plus-adapter growth, retaining only the best candidate. The canonical
reports are in
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_adapter_sharing_promoted_v1_2026-08-08/`.
The permutation control is mixed, so this does not establish order-invariant or
general continual learning.

Verifier-triggered local recovery then closed the tested permutation gap: the
same two seeds and source order `[2, 1, 0]` both passed after up to 128 extra
updates applied only to the newly grown capability. The strengthened bounded
promotion is archived in the same record; it demonstrates order robustness for
this control, while remaining short of general continual learning.

The same verifier-scored growth boundary now promotes four opaque capabilities
across seeds `69316` and `69317`, with two physical adapters serving four
logical bindings. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_adapter_sharing_four_promoted_v1_2026-08-08/`.
This remains a bounded four-capability result; unrestricted memory growth and
general lifelong learning are unproven.

The protocol then promotes five opaque capabilities across seeds `69316` and
`69317`, using two physical adapters for five logical bindings. Retention is
measured on a fixed held-out suite per mastered capability at every later
prefix. Reports are archived in
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_adapter_sharing_five_promoted_v1_2026-08-08/`.
This strengthens bounded continual growth; it does not establish unrestricted
lifelong learning.

The same contract now promotes six opaque capabilities across seeds `69316`
and `69317`, retaining only two physical adapters for six logical bindings.
Reports and accounting are archived in
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_adapter_sharing_six_promoted_v1_2026-08-08/`.
This is the current bounded scaling result; general lifelong learning remains
unproven.

The same contract now promotes seven opaque capabilities across both seeds,
with two physical adapters serving seven logical bindings. Reports and
accounting are archived in
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_adapter_sharing_seven_promoted_v1_2026-08-08/`.
This demonstrates reusable bounded growth, not unrestricted memory or general
lifelong learning.

The current pressure test reaches eight opaque capabilities across both seeds,
with two physical adapters serving eight logical bindings. Reports and
accounting are archived in
`session_records/sequence_working_memory_2026-08-02/learned_compute_candidate_adapter_sharing_eight_promoted_v1_2026-08-08/`.
This remains bounded reusable growth rather than general lifelong learning.

Growth admission now also tests fresh adapters against every protected compute
slot before allocating a new compute module. The matched audit preserved all
gates but selected fresh compute for the hard middle capability in both seeds,
so compute growth remains the next compression bottleneck.

The next shared-computation intervention attempted to consolidate four
protected procedures into one routed neural artifact. It is rejected: seed
`69317` passed, but seed `69316` failed at `0.699` and still failed at `0.738`
after doubling consolidation updates. The shared dense stack is therefore not
yet a reliable four-source consolidation mechanism. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/generated_composition_distilled_consolidation_four_source_rejected_v1_2026-08-08/`.

The stronger sequential consolidation path then passed the same four-source
pressure test on both seeds `69316` and `69317`. It adopted three successive
shared rewrites at each seed, each trained only on the newly arrived procedure
and retention-gated against all earlier aliases. After reload, all four opaque
procedures resolve to one shared artifact, source behaviors are
`0.9336 / 1.0000 / 1.0000 / 1.0000` and `0.9648 / 1.0000 / 1.0000 / 1.0000`,
reversal isolation and corruption rejection pass, and a held-out target reaches
mastery faster from the inherited artifact at both seeds. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/generated_composition_sequential_neural_consolidation_four_source_seed69316_promoted_v1_2026-08-08/`.

This promotes bounded replay-free sequential neural consolidation with
external capacity growth. It does not establish unrestricted memory growth,
arbitrary program induction, or general continual learning. The current
implementation is also too expensive for an always-on learning loop; reducing
verifier and optimization cost while preserving this retention contract is the
next high-ROI bottleneck.

The same sequential neural-consolidation boundary was then tested with five
runtime-generated eight-step opaque-rule procedures rather than hand-listed
primitive names. Across seeds `69316` and `69317`, all three shared rewrites,
four-source retention, exact reload, reversal recovery, corruption rejection,
frozen-core equality, and zero-replay gates passed. The inherited target was
never worse than fresh (`6,144 / 6,144` stable-prefix bits and `4,096 / 6,144`
bits respectively). Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/generated_composition_sequential_neural_consolidation_four_source_runtime_opaque_replicated_promoted_v1_2026-08-08/`.

This closes the hand-specified grammar gap for this bounded consolidation
mechanism, not the universal-computation gap. The exact audit required roughly
18 and 24 minutes per seed and 4,736 optimizer updates, so runtime-generated
procedure complexity and verification cost are now explicit implementation
bottlenecks.

## Promoted six-regime stable-prefix retention (2026-08-10)

The follow-up pressure test uses fresh unprotected challenger cells, eight
perturbed held-out contexts per regime, and delayed-feedback settlement before
declaring mastery. A held-out verifier qualifies a cell through
`verify_and_protect`; only then do generator content and route state freeze,
and the verified context prototype become an address prior. Relevant reversal
evidence can release the cell for a new challenger.

Seeds `85401` and `85402` both pass six-regime mastery, held-out qualification,
stable content, stable route, sparse materialization, shuffled-outcome,
missing-evidence, corruption, persistence, frozen-controller, and zero-replay
gates. Content floors are `0.9596`/`0.9617`; route floors are
`0.5884`/`0.6726`. Evidence is archived in
`session_records/policy_free_intention_prefix_growth_promoted_2026-08-10/`.

The claim is bounded stable-prefix retention, not positive transfer or general
continual learning. Fresh learners still win on some disjoint successors;
the next architectural pressure is verified factual/residual compute that
improves acquisition without copying policy-like generator weights.

## Promoted one-pass factual residual challenger (2026-08-10)

The factual challenger is now exercised end to end. A reusable source
transition model is frozen before a successor stream arrives. A separate
opaque context-addressed random-feature residual learns the successor delta
from `32` fresh transition rows in one pass. Admission is copy-on-write and
requires held-out one-step, recursive-rollout, and source-retention probes.

Both seeds `101` and `102` pass; the residual leaves the source model
byte-stable, while shuffled transition evidence is rejected and persistence is
exact. The matched full-model-copy and fresh controls fit the new regime only
after repeated optimizer updates and both fail source retention at target
stability. The promoted evidence is in
`session_records/policy_free_factual_residual_growth_promoted_2026-08-10/`.

This establishes a bounded factual-residual acquisition seam, not general
continual learning or unlimited computation. The next pressure test must
repeat residual growth across a longer bank and measure route cost,
compression, reversal, and memory-budget behavior.

## Promoted multi-regime factual residual stream (2026-08-10)

The longer pressure test admits six distinct factual regimes plus a reversal
into seven opaque residual slots. A reusable transition model is trained once
and frozen; each residual slot consumes `32` unique transition rows through
one-pass random-feature sufficient statistics. Copy-on-write admission
requires held-out one-step accuracy, a two-step recursive rollout, and
complete-prefix retention of all earlier slots.

Across seeds `101` and `102`, all seven lifetimes promote. Maximum retained
prefix MSE is `0.004544`/`0.009934`, maximum rollout MSE is
`0.018566`/`0.016057`, and source-retention MSE is `0.004217`/`0.000352`.
Opaque route round-trips recover slots `0..6`, requiring `21` existing-slot
comparisons across the seven novel bundles. The shared base remains
byte-stable; shuffled reversal evidence is rejected; missing and corrupted
evidence do not mutate committed memory; and persistence is exact.

Float16 compression passes held-out verification and reduces residual-bank
storage from `125,552` to `62,804` bytes. Int4 is rejected by the retention
probe. The residual path uses zero replay, while matched fresh controls use
`2,400` optimizer updates and replay `76,800` examples. Reports are archived
in `session_records/policy_free_factual_residual_stream_promoted_2026-08-10/`.

This promotes bounded factual-memory scaling and verified external growth, not
general continual learning, arbitrary new computation, or unrestricted memory
growth. The next boundary is a capacity-scaled residual basis with learned
route uncertainty, maintenance, and out-of-distribution retention controls.

## Promoted capacity-scaled factual memory and learned reliability (2026-08-10)

The next pressure test admits nine factual regimes plus a reversal into ten
opaque residual slots. The shared transition model remains frozen. After four
slots, a verifier-gated copy-on-write transaction expands capacity from `4` to
`8`; later admissions reach capacity `10` without changing retained content.
Each lifetime consumes `32` unique transition rows and passes held-out
one-step, recursive-rollout, and complete-prefix retention probes.

Across seeds `101` and `102`, all ten lifetimes promote. Maximum retained-prefix
MSE is `0.004544`/`0.014426`; route round-trips recover slots `0..9` after `45`
existing-slot comparisons. A rejected growth proposal is a no-op, shuffled
reversal is rejected, the frozen base remains byte-stable, and persistence is
exact.

An external replay-free error-bin reliability component consumes `142`
verifier outcomes without retaining rows. Clean reads score `0.917`/`0.958`
while corrupted and out-of-distribution evidence scores `0.250` for both
seeds. The learned gate allows clean reads and rejects the bad reads without
mutating residual or verifier state. Float16 compression reduces residual
storage from `179,360` to `89,720` bytes; int4 is rejected. Reports are in
`session_records/policy_free_factual_residual_capacity_promoted_2026-08-10/`.

This promotes bounded capacity-scaled factual memory and external learned
reliability, not general continual learning, arbitrary new computation, or
unrestricted memory growth. The next architectural boundary is a learned
procedure/capability stream under genuinely novel distributions, with
maintenance and verifier calibration still isolated from the controller.

## Trajectory-statistics route queries and automatic cell qualification (2026-08-10)

The memory-side route boundary now has an optional
`ExternalControllerTrajectoryQueryAdapter`. It keeps the factual planner on
the ordinary controller state, while an external router may address a growing
bank using the final learned controller representation plus masked mean/max
statistics of the learned event-token window. This is the direct import from
the exported games session's successful `trajectory_stats` direction. It is a
replaceable address query, not a modality-specific reasoning branch, and it
does not expose raw event formats or physical cell IDs to the controller.

`ExternalOutcomeIntentionRouter` also gives unqualified external cells a
differentiable exploration floor. An appended cell therefore receives causal
evidence before a learned route can suppress it permanently. The floor is
memory lifecycle behavior and remains separate from content generation,
retention, and decoder protocols.

Retention state is now tensor-only and versioned with context prototypes,
qualification counters, automatic protection, and hysteretic reversal. Low
outcomes from an unrelated opaque context do not release a protected cell;
only relevant evidence can start a reversal era. Legacy routed-memory payloads
migrate with a fresh retention ledger.

The six-regime pressure test in
`session_records/policy_free_intention_prefix_growth_rejected_2026-08-10/`
was rejected. The richer query and exploration floor did not overcome the
remaining policy-copy problem: a sampled intention generator is still a
policy-like artifact, so copying it into a new contradictory regime can cause
negative transfer. This confirms the exported session's plant/bank lesson:
reusable structure belongs in frozen computation, while new regimes should be
verified residual/delta candidates or fresh challengers selected
copy-on-write. Automatic protection also cannot be promoted from noisy
exploration rewards alone; it requires a held-out verifier prefix.

That verifier boundary is now executable through
`ExternalOutcomeIntentionRouter.verify_and_protect`. The caller supplies a
fresh outcome prefix and an opaque route context. If its minimum outcome
clears the configured floor, the transaction protects the cell and records
the qualification; otherwise the state is returned unchanged. The operation
does not update content, route logits, counters, or eligibility traces, so a
retention decision cannot accidentally become another learning update.

The runtime-opaque four-source audit was repeated in source order `[4, 3, 2,
0]` across seeds `69316` and `69317`. All rewrites, reload, retention,
reversal, corruption, frozen-core, and zero-replay gates passed. This promotes
order-robust retention for the bounded mechanism. Transfer efficiency remains
order-dependent: inherited/fresh target stable-prefix budgets were
`8,192 / 6,144` and `4,096 / 6,144` bits. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/generated_composition_sequential_neural_consolidation_four_source_runtime_opaque_permuted_replicated_promoted_v1_2026-08-08/`.

A matched throughput candidate doubled batch size to `32` and halved every
training budget while preserving nominal verifier-bit exposure. It failed at
the first growth stage (`0.6953` on the new source versus the `0.9375` gate),
despite parent stability, frozen-core, reload, corruption, and zero-replay
controls passing. Larger batches therefore cannot simply replace optimizer
updates in the current acquisition learner. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/generated_composition_sequential_neural_consolidation_batch32_same_bits_rejected_v1_2026-08-08/`.

A second cost intervention grouped independent retention probes into larger
recurrent batches. It preserved every semantic gate and produced identical
behavior, but increased wall time by `19.7%` (`1,129.8s` versus `944.1s`) on
the matched canonical audit. The implementation was reverted; evidence is in
`session_records/sequence_working_memory_2026-08-02/generated_composition_sequential_neural_consolidation_batched_probe_cost_rejected_v1_2026-08-08/`.

The stronger episodic context/credit boundary now extends the repeated-shift
schedule from 100 to 122 capabilities by adding a length-22 shift. Both seeds
pass causal credit, candidate permutation, old-route retention, full-bank
protection, reversal/recovery, persistence, corruption, reward-shuffled null,
and zero-replay gates. Targeted fresh remediation restores the final route
floor to `0.859375` on both seeds; seed `69317` briefly exposed a raw
length-22 floor of `0.640625` before remediation. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/repeated_shift_growth_6to8to10to12to14to16to18to20to22_remediated_v1_2026-08-08/`.

This promotes bounded 122-capability growth, not unbounded memory or general
continual learning. The next bottleneck is open-ended capability acquisition
with learned blueprint/compression or genuine new computation while retaining
the same no-replay and reversal guarantees.

The four-instruction register pressure test added a bounded residual operator:
normalized register state, bounded proposal, and opaque feature-wise gating.
It preserved all primitive-retention, null, missing-evidence, persistence,
corruption, frozen-core, and zero-replay controls across both seeds, and
improved serial execution stability. It did not improve transfer: inherited
composition required `20,480/24,576` bits versus fresh `16,384/8,192`; a
fresh-outcome blueprint-pretraining variant reached `0.9883/0.9688` final
composition but still required `8,192` bits versus fresh `4,096` on both
seeds. This rejects bounded normalization as a blueprint-reuse solution and
keeps held-out new-computation transfer as the next bottleneck. Evidence is
archived in
`session_records/sequence_working_memory_2026-08-02/external_register_four_instruction_bounded_residual_rejected_v1_2026-08-08/`.

The repeated-shift pressure test now reaches 144 capabilities through length
24. Large generated pattern banks use deterministic addressed-prefix
materialization rather than constructing the full combinatorial bank, so the
family namespace remains stable without a memory blow-up. Across seeds 69316
and 69317, all causal-credit, retention/reversal, persistence, corruption,
reward-shuffled, and zero-replay gates pass. Raw length-24 route floors were
`0.828125` and `0.796875`; targeted fresh remediation restored both final
new-route floors to `0.859375`. This is a promoted bounded scalability result,
not unrestricted memory growth, arbitrary new computation, or general
continual learning. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/repeated_shift_growth_6to8to10to12to14to16to18to20to22to24_prefix_bank_v1_2026-08-08/`.

The held-out procedure audit then froze four acquired opaque instruction
procedures and trained a fifth new code against fresh outcomes, compared with
a matched fresh interpreter. The valid depth-two rung passed all source,
target, retention, null, missing-evidence, reload, corruption, frozen-core,
and zero-replay controls, but inherited target acquisition required
`12,288` verifier bits versus `8,192` fresh on the repaired seed. This rejects
whole-procedure learning as blueprint reuse. The next higher-ROI test is a new
composition assembled from already learned opaque instruction data, which
tests reusable computation without asking one new code to encode an entire
unseen program. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_register_heldout_new_procedure_transfer_rejected_v1_2026-08-08/`.

The held-out composition intervention now produces the first replicated
positive transfer beyond whole-procedure storage. Four opaque instructions
are acquired sequentially, frozen, and reused in a new held-out order
`prefix_parity → complement → reverse → adjacent_xor`. Across seeds 69316 and
69317, inherited composition reaches stable mastery in `8,192` verifier bits
versus `12,288` for matched fresh learners (`1.5x` fresh-over-inherited).
Source retention, composition mastery, shuffled-outcome, missing-evidence,
reload, corruption, frozen-core, and zero-replay gates all pass. This promotes
bounded reusable compositional computation, not arbitrary program induction,
unrestricted memory growth, or general continual learning. The next bottleneck
is scaling the positive composition transfer across multiple held-out orders
and genuinely new primitive/operator combinations. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_register_heldout_composition_transfer_promoted_v1_2026-08-08/`.

The multi-order follow-up reuses the same four frozen opaque instructions in
three held-out orders. One seed transfers all three orders; the second
transfers two and ties fresh on the first even after doubling only composition
updates. All safety and retention controls pass, but the strict replicated
all-target gate is not promoted. This establishes order-robust partial
compositional reuse and localizes the next bottleneck to transfer calibration
across order distributions. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_register_multi_heldout_composition_rejected_v1_2026-08-08/`.

The strict held-out primitive test then froze the learned four-primitive
interpreter and acquired a fifth unseen `rotate` instruction. Both seeds pass
source mastery, target mastery, retention, null, missing-evidence, reload,
corruption, frozen-core, and zero-replay gates, but neither shows positive
transfer: one ties fresh and the other is slower. This cleanly separates the
current capability: reusable composition is promoted, while genuine new
primitive computation remains the largest architectural bottleneck. The next
intervention must learn a transferable operator family or code-to-computation
prior rather than add more instruction slots. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_register_heldout_primitive_transfer_rejected_v1_2026-08-08/`.

An anchored continual-blueprint control then allowed shared operator weights
to update on later source primitives while old instruction codes stayed
frozen. Anchor weights `1.0` and `10.0` produced fast new-primitive targets in
some seeds, but source capabilities fell below the hard mastery floor in every
run. This rejects naïve whole-blueprint updates and points to the next design
requirement: isolated meta-state or protected subspaces that can learn an
operator family without modifying mastered computation. Evidence is archived
in
`session_records/sequence_working_memory_2026-08-02/external_register_continual_blueprint_anchor_rejected_v1_2026-08-08/`.

The protected operator-family follow-up isolates a zero-initialized
code-conditioned residual while freezing the mastered base operator. It
preserves all source primitives but does not improve unseen `rotate` transfer;
mean initialization of the new code also ties or loses to fresh. The next
architecture requirement is therefore an explicit expandable computation
basis, not another fixed operator or code-space prior. Evidence is archived
in
`session_records/sequence_working_memory_2026-08-02/external_register_protected_meta_code_prior_rejected_v1_2026-08-08/`.

The first expandable-basis implementation adds an append-only
`ExternalRegisterComputeBasis` interface to the external register. Each slot
is independently addressable and versioned, consumes only the register plus
an opaque instruction vector, and can be trained or replaced without resizing
the controller or modifying earlier instruction data. The first unseen-
`rotate` audit added one fresh slot after the four source primitives were
mastered. It retained all sources, reached `0.9922` target accuracy, and
passed shuffled-outcome, missing-evidence, reload, corruption, frozen-parent,
and zero-replay controls. Stable-prefix promotion nevertheless failed against
the matched fresh learner (`8,192` stable bits), so this is an architectural
capacity/safety result rather than positive transfer. The remaining bottleneck
is learning when and how to reuse or compose expandable slots efficiently;
isolated growth alone does not yet make new computation cheaper than a fresh
learner. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_register_expandable_basis_probe_2026-08-08/`.

The register boundary now reuses the repository’s verifier-gated opaque
compute admission policy. Fresh probe outcomes can select an existing basis
slot only when every probe clears the mastery floor; otherwise memory-side
policy grows a slot. Mastered slots are independently frozen, and the newest
unpromoted slot can be rolled back. This keeps capacity growth reversible and
prevents an unverified candidate from silently becoming shared computation.

The first two-seed basis-reuse audit now promotes the next boundary. After a
`rotate` basis slot was mastered and frozen, a fresh opaque instruction reused
that slot from fresh outcomes only. Reuse reached `4,096` stable verifier bits
on both seeds, versus `12,288` and `8,192` for matched fresh learners. The
slot digest, source capability, shuffled-outcome, missing-evidence, reload,
frozen-parent, and zero-replay gates all passed. This is bounded sample-
efficient reuse of mastered external computation; general new-primitive
induction and unrestricted continual learning remain open.

The distinct-operator follow-up reused the frozen `rotate` basis for a fresh
`global_parity` instruction. It passed safety and retention on both seeds, but
stable transfer was asymmetric: one seed improved (`4,096` versus `8,192`
fresh bits), while the other regressed (`16,384` versus `8,192`). Strict
cross-operator promotion is rejected. The next bottleneck is learned
compatibility/routing across primitive families, not external-slot growth or
same-family reuse.

The executable route-vs-grow audit now passes both seeds. When efficiency
admission rejected reuse, the system appended and trained a new slot: seed
`69316` recovered `8,192` stable bits on slot 1 while retaining the old
capability at `1.0000`. When admission accepted reuse, seed `69317` used slot
0 at `4,096` stable bits and also retained the old capability at `1.0000`.
The old basis digest remained unchanged in both cases. This promotes
efficiency-aware reversible capacity routing for the tested boundary, not a
general learned cross-operator prior.

Efficiency-aware basis admission now requires both fresh mastery and a stable
cost no worse than the matched fresh learner. In the cross-operator rerun it
correctly requested growth for the slower seed (`16,384` versus `8,192` fresh)
and reused the basis for the faster seed (`4,096` versus `8,192`). This closes
the correctness-only admission gap while preserving opaque memory-side
selection. The executable grow branch then recovered the slower case without
affecting the mastered slot, completing this route-vs-grow boundary.

The register now also exposes an `ExternalRegisterBasisCompatibilityPrior`.
It reuses the existing learned opaque candidate-screen mechanism over learned
slot signatures and instruction-vector queries, training only from attempted
scalar outcomes. It is deliberately a screening prior rather than an
admission authority: fresh stable-prefix verification remains mandatory.

The opaque compatibility prior now has a two-seed held-out screening audit.
It reduced mean candidate trials on admissible queries from `2.074` to `1.116`
and from `2.143` to `1.029`, while preserving exact verifier admissibility and
using zero replay. This promotes trial-order efficiency only; fresh stable
verification remains mandatory for reuse or growth decisions. Evidence is
archived in
`session_records/sequence_working_memory_2026-08-02/external_register_compatibility_prior_audit_2026-08-08/`.

The real multi-slot acquisition follow-up trained three source primitives into
independent basis slots, updated the opaque prior from their actual verifier
outcome matrix, and routed held-out `prefix_parity` through the live register
scheduler. Both seeds correctly requested growth because no existing slot
passed fresh verification, with zero replay. This establishes no-false-
admission behavior during real acquisition; positive transfer to a genuinely
new primitive remains open.

Executing the selected growth slot reached high final target accuracy and
retained all source capabilities with unchanged old basis digests, but both
seeds failed stable-prefix promotion and failed the shuffled-outcome rejection
control. The growth result is rejected. This localizes the next bottleneck to
causal credit/verification dependence during new-slot acquisition, not
append-only capacity or retention. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_register_real_basis_acquisition_2026-08-08/`.

The causal repair changes only new-slot optimization to attempted-outcome BCE,
which exposes delivered scalar verifier outcomes instead of verifier-private
correct-action utilities. Shuffled-training controls collapse to `0.4766` and
`0.5000`; normal target accuracy remains `0.9375` and `0.9063` with source
retention intact. Stable-prefix promotion still fails, so growth remains
rejected while the credit-path repair is retained.

The next concurrent audit promotes a stronger boundary: two fresh direct
capabilities and a fresh decoder/bridge for a frozen three-instruction source
program learned in the same round-robin schedule. Across seeds 69316 and
69317, all three candidates passed frozen consolidation probes, missing-
evidence and shuffled-outcome controls, and exact fixed-suite source
retention, with zero replay. This is bounded concurrent compositional reuse,
not unrestricted program induction or general continual learning. Evidence is
archived in
`session_records/sequence_working_memory_2026-08-02/external_register_real_basis_acquisition_2026-08-08/interleaved_composition_transfer/`.

The subsequent replicated audit interleaved two independently ordered frozen
source programs with the same two direct capabilities. All four fresh
decoder/bridge candidates passed consolidation, missing-evidence,
shuffled-outcome, and exact source-retention gates across both seeds with zero
replay. This is stronger bounded concurrent compositional reuse, while larger
program grammars, fresh-learner transfer, and unrestricted continual learning
remain open. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_register_real_basis_acquisition_2026-08-08/interleaved_multi_composition_transfer/`.

The full finite three-source permutation grammar has now also passed the
concurrent audit. Six independently ordered frozen programs and two direct
capabilities all passed consolidation, causal, missing-evidence, and exact
source-retention gates across both seeds with zero replay. This remains a
bounded grammar result rather than unrestricted program induction or general
continual learning. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_register_real_basis_acquisition_2026-08-08/interleaved_full_composition_grammar/`.

The matched fresh-learner arm is now part of the full-grammar audit. It does
not yet promote positive sample-efficiency transfer: the inherited path
passes its behavior gates, but fresh stable-prefix outcomes are incomplete or
mixed across seeds. This identifies decoder/event-bridge adaptation as the
next bottleneck and preserves the distinction between capability retention
and faster learning. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_register_real_basis_acquisition_2026-08-08/interleaved_full_grammar_transfer/`.

Raw reuse of a mastered action decoder was tested as the next adaptation
shortcut and rejected after seed- and program-dependent effects. The
production boundary therefore does not assume decoder weights transfer across
skills; a future interface prior must be protocol-agnostic and independently
verified. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_register_real_basis_acquisition_2026-08-08/interleaved_decoder_prior_diagnostic/`.

A shared event bridge trained from mastered source outcomes was tested as a
protocol-agnostic interface prior. It preserved source retention but failed
the replicated composition gate in one seed, so a single frozen bridge is not
the production solution. Future interface adaptation must remain capability-
conditioned and independently verified. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_register_real_basis_acquisition_2026-08-08/interleaved_shared_bridge_prior_diagnostic/`.

A capability-conditioned event bridge now provides an opt-in reusable
interface prior. It conditions shared bridge weights on opaque learned
program vectors rather than copying protocol-specific decoder weights. One
seed passed the composition and transfer gates, but the second failed both
composition candidates, so the mechanism is not promoted yet. Evidence is
archived in
`session_records/sequence_working_memory_2026-08-02/external_register_real_basis_acquisition_2026-08-08/interleaved_conditioned_bridge_prior_diagnostic/`.

Repeating source acquisition with two restart candidates repairs the
conditioned-bridge behavior rung across both seeds. This promotes bounded
source-robust compositional retention, while matched fresh learners still do
not establish positive sample-efficiency transfer. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_register_real_basis_acquisition_2026-08-08/interleaved_conditioned_bridge_restart_repair/`.

The fresh transfer control now acquires the same source primitives before
target composition learning. Fresh learners reached stable target mastery in
8,192 verifier bits for both programs across both seeds, while inherited paths
were slower in three of four comparisons. This rejects positive transfer and
localizes the bottleneck to source-state geometry/interface alignment rather
than memory capacity. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_register_real_basis_acquisition_2026-08-08/interleaved_conditioned_curriculum_transfer/`.

## Staged outcome-only bridge necessity diagnostic (2026-08-09)

The export-session lesson that necessity pressure should follow acquisition was
implemented as an opt-in bridge-prior stage. A frozen parent and source
decoders first trained a capability-conditioned event bridge from sampled
scalar verifier outcomes on valid evidence. A second stage then trained only
the bridge from sampled scalar outcomes on norm-matched unusable evidence while
masking the frozen controller state. The candidate selector retained an
acquisition checkpoint whenever the necessity phase damaged source behavior.

The two-seed diagnostic preserved source and target behavior gates, but it did
not improve the alignment bottleneck or positive transfer. On seed `69316`,
the necessity phase ended with source scores `0.770/0.777/0.699` and decoy
scores near `0.46–0.55`; on seed `69317`, source scores ended at
`1.000/0.898/0.945` and decoy scores worsened to `0.605/0.539/0.559`.
The transactional selector therefore kept the earlier acquisition state.
Matched fresh target prefixes were incomplete in this reduced audit, so no
sample-efficiency claim is made.

This rejects necessity pressure as the missing bridge-transfer mechanism. It
does retain two useful rules: acquisition and protection must be separate
phases, and a candidate must be rejected if protection damages source state.
The canonical path should not add this loss by default. The unresolved
bottleneck remains learned source-state/interface geometry, which requires a
capability-conditioned alignment signal or a richer factual external model,
not a stronger default-suppression penalty. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_register_staged_bridge_necessity_rejected_2026-08-09/`.

An optional generic normalized program-counter signal was then tested in the
shared register interpreter. It improved primitive retention on one matched
four-instruction rung (`0.8828`) but the inherited composition never reached a
stable prefix and finished at `0.7148`, below the no-position baseline's
`0.8477`; the fresh learner reached stable mastery in `16,384` verifier bits.
The position injection is rejected as a current transfer accelerator and was
removed from the production path. The result does not disprove a future
execution context, but shows that adding structural position to the existing
instruction code does not repair the portable serial algebra. Evidence is in
`session_records/sequence_working_memory_2026-08-02/external_register_execution_position_rejected_v1_2026-08-09/`.

## Portable external-program artifact contract (2026-08-09)

The external-program boundary now has a typed, independently versioned file
contract: `ExternalProgramArtifact`. It contains only finite learned opaque
instruction tensors and the interpreter, execution, and optional output
interface schemas needed to validate a compatible runtime. It has no task
name, modality identity, protocol action, correct answer, or verifier-private
metadata. Artifacts produce a deterministic interface-and-content digest,
round-trip through a torch-safe payload, reject tensor tampering, and can be
admitted to or snapshotted from `ExternalSequenceProgramMemory` only after ABI
validation.

This closes a real implementation seam between “external memory as files” and
the register interpreter, but it is an interface/persistence gain rather than
evidence of new learned capability or positive transfer. Behavior still must
be verified after loading, and the next capability bottleneck remains a
portable execution-state representation shared across longer programs.

The runtime now exposes that state boundary explicitly through
`ExternalExecutionSnapshot`. It separates the durable observed register from
the transient executed register and opaque intermediate trace, carries an
optional program-artifact digest, and round-trips through a versioned
tensor-only payload. Existing `read_execute_register()` behavior remains
compatible; the typed snapshot is the contract for new callers. This is a
correctness and replacement seam, not yet a claim that the learned serial
interpreter can solve arbitrary longer programs.

## Shared learned operator basis diagnostic (2026-08-09)

The register interpreter now has an opt-in
`factorized_shared_operator_basis` mode. Instead of giving each opaque
instruction code an effectively independent transition, the mode learns a
small common basis of low-rank state-transition factors. An instruction code
selects a mixture over those factors and a bounded composition gate applies the
result to the learned register. The controller still sees only learned opaque
instruction and state tensors; no task identity, protocol action, execution
position, correct answer, or verifier-private metadata is introduced.

The matched full screen used two seeds, three mastered source primitives, two
held-out direct primitives, and two held-out three-instruction compositions.
All `8/8` target gates passed across the two seeds, every source retention delta
was exactly zero, and shuffled-training, missing-evidence, persistence,
corruption, and frozen-parent controls passed. This promotes a narrow
behavior/retention result: a shared transition algebra can support the tested
compositions without overwriting the sources.

It does not yet promote general continual-learning transfer. Fresh positive
transfer occurred in only `1/4` strict stable-prefix comparisons (`1/2` on the
first seed and `0/2` on the replication). The mode is therefore retained as a
qualified opt-in architectural direction, not made the default. The next test
must target portable execution-state algebra and genuinely new longer
programs, with no-replay fresh-learning curves and retention controls. Evidence
is archived in
`session_records/sequence_working_memory_2026-08-02/external_register_shared_operator_basis_full_transfer_2026-08-09/`.

## Shared operator-program depth rejection (2026-08-09)

A temporary follow-up gave each opaque instruction a two-step latent program
over common transition factors. This was intended to make new computation a
composition of reusable atomic transitions while keeping the controller
frozen. The matched audit used the same `576` source updates and `512` target
updates as the shared-basis screen.

The result was negative. Seed `69316` passed target behavior and exact source
retention, but inherited learning was slower than fresh on both compositions:
`40,960 > 24,576` and `24,576 > 16,384` stable verifier bits. Seed `69317`
failed one composition behavior gate (`0.7422`) and had no inherited stable
prefix for it; the other composition required `40,960 > 16,384` fresh bits.

The branch was removed rather than retained as another public experimental
option. The evidence shows that extra latent execution depth, without a
learned adaptation rule, increases optimization burden. The next architectural
frontier is protected meta-plasticity: an external, independently versioned
update state trained explicitly for rapid new-capability adaptation while
mastered computation remains frozen. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_register_shared_operator_program_rejected_2026-08-09/`.

## External fast-weight plasticity primitive (2026-08-09)

The memory boundary now exposes `ExternalFastWeightPlasticity`, an
independently versioned outcome-gated delta rule. It reads learned opaque
query/value tensors and updates an external per-capability matrix state; the
controller and the plasticity-rule parameters do not change during the
acquisition stream. Failed outcomes and missing evidence leave the
computation-state weights unchanged, while a positive outcome writes toward
the observed value.

Two seeds passed the bounded pressure test: source and target associations
reached the stable readout gate in one verifier bit, source state was retained
exactly while target state was acquired, persistence was exact, and zero replay
was used. This promotes an isolated external associative-plasticity primitive,
not general continual learning or arbitrary new computation. The current
target starts from a fresh state, so positive transfer has not been tested.

The next high-ROI boundary is to connect this state to a learned capability
adapter and run a matched fresh-learner curve. That test must establish that
the external state makes a genuinely new capability faster to learn while
protected source capabilities remain unchanged. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_fast_plasticity_promoted_2026-08-09/`.

## External fast-weight capability-adapter transfer (2026-08-09)

`ExternalFastWeightCapabilityProgram` now connects the external fast-weight
state to the intention bus through a replaceable learned adapter. The frozen
processor provides learned event and intention tensors; the memory-side query
encoder addresses one external state, and only a positive outcome can write an
opaque attempted action. A read is transformed into an intention residual. No
task identity, correct unattempted action, raw modality, or device protocol is
introduced.

The matched two-seed audit trained the shared intention adapter on `64` unique
source lifetimes, froze the entire inherited program, and then acquired `16`
fresh target lifetimes with zero target optimizer updates and zero source
replay. The inherited target reached the stable `0.95` cosine floor after `1`
target example on both seeds. A fresh adapter receiving the same target stream
needed `7` and `14` examples respectively. Source retention floors were
`0.9971` and `0.9979`; failed-outcome no-write, missing-evidence no-write,
state persistence, frozen-parameter, and accounting controls passed.

This promotes a qualified interface-prior transfer result: an isolated
growable memory state can reuse a learned intention mapping when a new
capability is instantiated. It is not general continual learning. The audit
uses a deliberately regular action-to-intention relation and does not test
variable-length working-memory programs, delayed credit assignment,
interference between many states, learned write/eviction/compression, or
arbitrary new computation. The next pressure test should vary the relation and
require a new multi-step capability to be acquired from outcome-only feedback
while source traces remain protected.

Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_fast_plasticity_capability_transfer_2026-08-09/`.

## External outcome-credit eligibility state (2026-08-09)

The memory boundary now exposes `ExternalOutcomeCreditPlasticity` and
`ExternalOutcomeCreditState`. The state contains an external capability policy,
its decayed log-probability eligibility trace, a scalar baseline, and explicit
decision/feedback counters. It consumes only learned feature tensors, opaque
sampled choices, exact logging propensities, presence, terminal markers, and
deterministic scalar outcomes. Delayed feedback updates the external policy and
terminal feedback clears transient eligibility without touching the frozen
controller or plasticity-rule parameters.

The two-seed pressure test used a hidden two-step event-to-choice relation. It
trained `2,000` source episodes, then `5,000` fresh target episodes from one
terminal verifier bit per episode, with `500` held-out evaluations at each
`500`-episode prefix. Trace-enabled target accuracy reached `0.980` and
`0.972`; the stable `0.90` gate was reached after `500` episodes on both
seeds. The matched no-trace controls remained at `0.518` and `0.470`, leaving
the first decision at chance, while reward-shuffled controls fell to `0.368`
and `0.194`. Source capabilities retained `0.958` on both seeds.

All causal, missing-feedback no-write, persistence, frozen-rule, and zero-
replay gates passed. This promotes a bounded delayed-credit primitive and is a
stronger learning result than the earlier supervised interface-adapter audit.
It is still not general continual learning: the policy is fixed-width, the
relation is linear in learned event features, and the test does not yet cover
variable-length programs, many-capability interference, learned memory
allocation, or consolidation. The next integration must connect this credit
state to executable external programs and test a genuinely new multi-step
capability across a changed relation while preserving old states.

Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_outcome_credit_promoted_2026-08-09/`.

## External outcome-credit sequence-length scaling (2026-08-09)

The eligibility state was generalized from two to an arbitrary fixed number of
temporal phases without changing its update rule or exposing verifier labels.
The matched three-phase rung used `3,000` source and `4,000` target episodes per
seed, one terminal scalar outcome per episode, and no replay. Target sequence
accuracy was `0.9567` and `0.9733`; stable `0.90` mastery was reached at
`1,500` and `1,000` target episodes. The no-trace controls reached only
`0.1133` and `0.3800`, reward-shuffled controls `0.1267` and `0.1533`, and
source retention remained `0.9400` and `0.9267`.

This promotes replicated three-phase delayed-credit scaling. Four phases show
the expected causal signal but are not promoted yet: short exploratory rungs
either leave source mastery below the gate or fail the stable target prefix.
The next bottleneck is variance and acquisition cost under sparse terminal
feedback, not basic eligibility correctness. The next intervention should test
a learned external value baseline or variance-reduced credit, with the same
source-retention and shuffled-outcome gates.

Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_outcome_credit_three_phase_promoted_2026-08-09/`.

## External outcome-credit four-phase value baseline (2026-08-09)

The external delayed-credit policy now has an optional external,
feature-conditioned value baseline. The baseline is trained only from learned
features and terminal scalar outcomes, and supplies a trajectory baseline to
the policy update. Its weights, feature trace, prediction trace, and counters
are external tensor state; neither the frozen controller nor the plasticity
rules are updated. Missing feedback remains a no-write path, and the state
round-trips through a versioned tensor payload.

The matched four-phase audit used `3,000` source and `3,000` target episodes
per seed. With the baseline enabled, target exact-sequence accuracy reached
`0.9067` and `0.9033`, with stable `0.90` prefixes at `2,500` and `3,000`
episodes. Source retention was `0.9233` and `0.9067`; no-trace controls were
`0.2067` and `0.0167`, and reward-shuffled controls were `0.0833` and
`0.0233`. Both seeds used zero replay and zero optimizer updates and passed
the source, stability, causal-control, missing-evidence, persistence, and
frozen-rule gates.

This promotes four-phase delayed scalar credit with external value-baseline
variance reduction. It remains a bounded continual-memory primitive rather
than general continual learning: the policy is fixed-width, the relation is
linear in learned event features, and the audit does not test changing
relations, many-capability interference, arbitrary executable programs, or
unrestricted memory growth. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_outcome_credit_four_phase_value_baseline_promoted_2026-08-09/`.

## Outcome-only routing over executable external programs (2026-08-09)

The memory boundary now exposes `ExternalOutcomeProgramRouter` and
`ExternalOutcomeProgramRouterState`. The router is an append-only choice
layer over a separately versioned executable-program memory: it masks
unadmitted candidates, samples opaque program indices, records exact behavior
propensities, carries eligibility across a sequence, and applies one terminal
scalar verifier outcome to the external route state. Activating a new program
does not resize the controller or change the learned program rule. Router
state includes the active candidate count and round-trips through a nested
tensor-only payload.

The matched audit used three pre-admitted artifacts executed by the shared
external register interpreter. Two programs were available during source
acquisition; a third was activated before a fresh two-phase target relation.
The controller, interpreter, program memory, router rule, and value-baseline
rule were frozen during route learning. Target accuracy was `0.9100` and
`0.9300`, with stable prefixes at `8,500` and `3,500` target episodes. Source
retention was `0.9800` and `0.9767`. No-trace controls reached `0.4300` and
`0.2100`, reward-shuffled controls `0.0733` and `0.1300`, and the no-append
capacity controls `0.5767` and `0.3967`. Both seeds passed appended-program
use, missing-evidence, persistence, frozen-component, source-state isolation,
and zero-replay gates; route acquisition made zero optimizer updates.

This promotes a bounded bridge from scalar delayed credit to executable
external programs. It is not program induction, arbitrary new computation,
unrestricted growth, or general continual learning: the executable artifact
interpreter is pre-admitted, the route capacity is finite, and source and
target capability states are isolated. The next bottleneck is interference
and transfer when many capabilities share one growing address space, followed
by fresh-relation transfer against a matched fresh executor. Evidence is
archived in
`session_records/sequence_working_memory_2026-08-02/external_outcome_program_router_promoted_2026-08-09/`.

## External transition memory and model-based search (2026-08-09)

The production package now exposes `ExternalTransitionModel`,
`ExternalTransitionObservation`, and `ExternalModelBasedPlanner`. The model
is an independently replaceable external component that learns only from
opaque state/intention/next-state tensors. The planner derives an intention
sequence by searching candidate opaque intentions against an opaque terminal
goal state or a runtime-sized set of possible goal states; it does not store a
task-specific policy and does not resize the controller when candidate or goal
count changes. When an `ExternalGoalEvaluator` is supplied, a goal set is an
existential opaque predicate: the best verifier score for any member is the
terminal objective. This supports goal fragments without assigning semantics
to latent coordinates.

The planner also exposes an active disambiguation primitive. When several
factual slots remain compatible with the current evidence, it can select the
opaque intention whose predicted consequences disagree most across those
slots. The caller executes that intention and routes the observed consequence
through the ordinary verifier; no task policy or slot label is injected. This
is the model-based form of probe addressing. It was initially retained as a
diagnostic capability pending a causal probe-versus-random control.
The online context router now exposes the same operation as a read-only
request, deriving plausible stable slot IDs from the opaque evidence window
when the caller does not supply them.

That narrow causal control now passes across seeds `83001`, `83002`, and
`83003`: active model-disagreement probing routed the hidden regime at `1.0`,
while uniform random intention selection reached `0.750`, `0.773`, and
`0.730`. The controller and bank were unchanged during queries and persisted
exactly. This qualifies active factual disambiguation for the two-regime
synthetic boundary only; learned probe selection, noisy outcomes, multimodal
usefulness, and general continual learning remain open. Evidence is archived
in
`session_records/sequence_working_memory_2026-08-02/external_model_disambiguation_probe_promoted_2026-08-10/`.

A follow-up widened the candidate intention set to eight and added Gaussian
outcome noise (`0.1`). Across seeds `83101`, `83102`, and `83103`, active
probe routing reached `0.984`, `0.977`, and `0.980`, against random controls
at `0.801`, `0.820`, and `0.754`. All seeds passed the predeclared `0.95`
quality gate, selected the informative intention, preserved the frozen
controller and bank, and restored exactly. This strengthens the bounded probe
result; live asynchronous-router integration and learned probe selection
remain open. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_model_disambiguation_probe_wide_noisy_promoted_2026-08-10/`.

The follow-up online seam is now promoted across seeds `83201`, `83202`, and
`83203`. Each run sent an ambiguous opaque evidence window through the
router's read-only probe request, executed the returned intention in a hidden
regime, and submitted the noisy consequence through the ordinary observation
path. Active resolution and routing were `1.0` on every seed, versus random
resolution of `0.145`, `0.117`, and `0.137`. The audit used eight opaque
one-hot candidate intentions, outcome noise `0.1`, and fixed factual matching
tolerance/margin `0.5`/`0.05`; the controller and factual bank were unchanged,
and router persistence was exact. This qualifies the request--execute--observe
integration only: the candidate basis and factual models are fixture-supplied,
so learned probe formation, multimodal usefulness, irreversible dynamics, and
general continual learning remain open. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_online_disambiguation_probe_promoted_2026-08-10/`.

The first matched two-seed pressure test froze the controller, trained the
external model for `1,200` source updates, then acquired three target goals
with zero target optimizer updates and zero replay. All target and retention
gates reached `1.0`; shuffled-goal and shuffled-transition controls reached
`0.0`; fresh-model controls reached `0.3333` and `0.0`; persistence and frozen
controller checks passed.

This is the first repository rung that directly tests the exported session's
strongest architectural result: store factual transition knowledge externally
and compute behavior from the current goal. The initial smoke also exposed an
important boundary rule: Euclidean progress in an opaque latent space is not
valid by default. The promoted planner uses terminal opaque-goal matching;
future general planners need an independently learned/verifier-grounded goal
evaluator rather than hand-assigned latent geometry.

The result is bounded. The fixture is deterministic and small, the goals use
one dynamics family, and the model is not yet tested on disjoint dynamics,
learned goal abstraction, compression, or unrestricted growth. The next
pressure test must vary dynamics, retain earlier transition knowledge, report
actual model updates and search compute separately, and compare against a
matched fresh model. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_model_based_planner_initial_rung_2026-08-09/`.

## Opaque cost-aware goal search (2026-08-10)

The planner now accepts runtime-sized nonnegative opaque intention costs and
can minimize terminal goal error plus accumulated step cost. This implements a
key lesson from the exported learning session: optimizing immediate success
alone does not pressure the system to retrieve and reuse an existing solution;
the long-horizon objective must measure the cost of reaching the goal.

The three-seed pressure test learned one affine factual model from fifteen
opaque transition rows. Terminal-only and cost-aware search both reached the
same goal, but cost-aware search reduced route cost from `10` to `1` on every
seed. Shuffling the cost vector changed the route and failed the goal control.
The controller and factual model were unchanged during search, persistence
was exact, and planner search performed zero optimizer updates or replay.
This qualifies cost-aware inference only: cost prediction, irreversible
dynamics, retrieval-before-learning across a growing bank, and general
continual learning remain open. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_cost_aware_planning_promoted_2026-08-10/`.

## Replay-free learned-cost search with an irreversible trap (2026-08-10)

The cost boundary now survives a stronger three-seed fixture with an
absorbing trap. A random-feature factual model learned the opaque transition
function once, while a separate affine external model learned scalar
intention costs from the same one-pass evidence. Cost-aware search reached the
goal on every seed, avoided the irreversible trap, and reduced actual route
cost from `5/10/6` under terminal-only search to `2/2/2`. Shuffled cost
controls changed behavior and failed the goal. Both external models and the
controller remained unchanged during planning; persistence, quality, and
zero-replay gates passed.

This promotes a bounded replay-free learned-cost planning seam, not general
irreversible-world competence: the scalar cost model is simple, the trap
fixture is tiny, the horizon is finite, and cost/address learning across a
growing bank remains open. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_irreversible_cost_planning_promoted_2026-08-10/`.

## Multi-slot cost-aware factual retrieval (2026-08-10)

The cost boundary now extends across three persistent factual slots with
different opaque transition scales. For each of three opaque goal queries, the
planner searched every slot over a three-step horizon without receiving a
task/context label. A separately learned affine scalar-cost model supplied
opaque intention costs to the same search.

Across seeds `83321`, `83322`, and `83323`, cost-aware search reached all goals,
selected stable slot IDs `[0, 1, 2]`, and reduced realized total route cost
from `18` under terminal-only search to `9`. The controller, factual bank, and
cost model remained unchanged during search; all `45` factual and `45` scalar
cost rows were consumed once, with zero replay and exact persistence.

This promotes bounded multi-slot factual retrieval and cost-aware planning.
It does not establish learned address formation, nonlinear model growth,
compression, unrestricted memory growth, or general continual learning.
Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_multi_slot_cost_selection_promoted_2026-08-10/`.

## Append-only transition memory across disjoint dynamics (2026-08-09)

The next two-seed rung adds `ExternalTransitionMemory`, an append-only factual
store keyed by opaque state, intention, and context tensors, plus
`ExternalGoalEvaluator`, a replaceable scalar verifier trained from outcome
bits. The same state/intention pairs are presented under two different opaque
contexts with different dynamics. The target phase appends 12 target facts to
12 source facts; it performs zero target optimizer updates and replays zero
source examples.

Both seeds reached `1.0` target mastery and `1.0` source retention after the
target append. Goal/context shuffles and corrupted-memory controls rejected,
fresh-memory controls remained below mastery, persistence was exact, and the
controller digest was unchanged. This supports the narrower claim that
external append-only factual memory can prevent interference between two
explicitly separated dynamics regimes while behavior is recomputed by search.

The result remains bounded nonparametric memory, not general continual
learning. Context discovery is supplied by the fixture, transitions are exact
stored matches, and there is no learned consolidation, compression, or
extrapolation beyond stored facts. The next bottleneck is to learn when and how
to create a context/address, then test many regimes with controlled capacity
growth and compression. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_transition_memory_transfer_promoted_2026-08-09/`.

## Learned context-address admission (2026-08-09)

The next two-seed rung removes the supplied context labels. A replaceable
`ExternalContextAddressResolver` receives only complete opaque transition
bundles and verified next-state tensors. It reuses an existing context address
only when the append-only transition facts explain every row within tolerance;
otherwise it allocates a fresh opaque handle. The controller still receives
only learned tensors, and the resolver remains outside the controller.

Each seed presented three unique dynamics regimes and one duplicate regime.
The resolver discovered exactly three addresses, reused the duplicate address,
and retained all regimes—including a reversal—with `1.0` behavioral mastery.
Wrong-context bundles produced non-zero factual next-state error, fresh memory
produced zero factual hits, corrupted facts were rejected, persistence was
exact, and the controller remained unchanged. The run used 48 transition
lifetimes, 12 verifier bits, zero target optimizer updates, and zero replay.

The direct factual controls are stronger than the behavioral shuffle result:
some wrong-context plans reached individual goals by chance, so goal mastery
alone is not treated as evidence of context causality. The promoted result is
therefore only that verified transition consistency can drive bounded opaque
address admission and preserve multiple stored dynamics. It is not yet
general continual learning: admission consumes a complete bundle, uses exact
stored facts, and allocates opaque handles rather than learning partial-stream
clustering. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_context_address_transfer_promoted_2026-08-09/`.

## Online partial-evidence context admission (2026-08-09)

The next two-seed rung extends address admission from complete bundles to
interleaved online evidence. `ExternalOnlineContextAddressResolver` keeps
provisional rows outside the transition store until three consistent verified
observations have accumulated. For an already-bound opaque stream, a first
contradiction returns `conflict` with zero writes; a second contradiction
admits a new address and commits only the buffered contradictory facts.

Both seeds passed the no-early-write, interleaved-admission, duplicate-reuse,
retention, reversal, wrong-context factual-error, corruption, fresh-memory,
persistence, and frozen-controller gates. The first six interleaved
observations produced zero memory records; three addresses were then admitted,
and the reversal created a fourth without changing the original facts. The
target phase used zero optimizer updates and zero replay.

This is still a bounded memory-side protocol, not general continual learning.
The stream binding is opaque transport state, thresholds are fixed, and the
resolver does not yet infer contexts from raw modalities, learn its evidence
thresholds, cluster arbitrary partial evidence, or compress unbounded history.
Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_online_context_admission_promoted_2026-08-09/`.

## Learned transition evidence and read-only reuse (2026-08-09)

The next two-seed rung adds `ExternalTransitionEvidenceEvaluator` as an
independently trainable memory-side verifier. It classifies whether a stored
opaque next-state prediction remains consistent with a noisy observed tensor.
When reuse is accepted, the online resolver is read-only: it does not replace
the mastered factual row with the noisy observation. Only genuinely admitted
new evidence writes to memory.

Both seeds accepted a noisy duplicate as reuse with zero writes and exact
source retention, while the fixed exact-match control allocated a duplicate
context. Contradictory evidence still admitted a new context; wrong-context,
fresh-memory, persistence, and frozen-controller gates passed. Target
adaptation used zero optimizer updates and zero replayed examples.

The evaluator's own pretraining cost is accounted for explicitly: 1,024
synthetic verifier rows, 500 optimizer updates, and 510,976 repeated training
rows. This promotes a robustness boundary only; it is not replay-free learning
of the evaluator. The next bottleneck is to learn evidence calibration online
from real multimodal event tensors and test whether the verifier transfers
across representation changes without replay. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_learned_evidence_admission_promoted_2026-08-09/`.

## Context-isolated external calibration (2026-08-09)

The next two-seed rung isolates online adaptation in external memory. A frozen
`ExternalTransitionEvidenceEvaluator` supplies the reusable evidence score,
while `ExternalContextualEvidenceCalibrator` maintains append-only scalar
temperature and bias states keyed by opaque context tensors. The resolver can
pass a candidate context through the same evidence boundary; the controller
and evaluator remain frozen.

Each target stream received one deterministic verifier outcome at a time. The
target held-out accuracy improved from `0.477`/`0.496` to `0.926`/`0.928` across
the two seeds, while source accuracy stayed at `1.0`. The source slot and base
evaluator were byte-stable, a wrong-context control did not obtain the gain,
and the external calibration payload restored exactly. The target phase used
256 unique verifier bits, 256 optimizer updates, and zero replayed target
examples per seed.

This is the first promoted memory-side update that adapts online without
replaying old target examples while retaining an isolated prior capability.
It remains bounded continual calibration, not general continual learning:
context vectors are supplied, scalar adaptation cannot add arbitrary
computation, slots grow append-only, and there is no learned context discovery,
consolidation, compression, or unbounded-stream test. The next bottleneck is
to learn context/address formation and test many alternating regimes under
capacity pressure. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_contextual_calibration_promoted_2026-08-09/`.

## Policy-free model-bank acquisition on disjoint dynamics (2026-08-09)

The next two-seed rung follows the exported session's model-versus-policy
finding. `ExternalTransitionModelBank` stores append-only opaque transition
models outside the controller. A new target slot is initialized from the
source model, then trained in isolation. `ExternalModelBasedPlanner` derives
behavior by searching candidate intentions against the current opaque goal;
there is no task-specific policy in the controller or bank.

On two disjoint dynamics regimes, early stopping at the same transition-loss
threshold required 23 versus 35 updates and 29 versus 36 updates for
source-initialized versus matched fresh target models. Both target models
reached `1.0` mastery in both seeds. Source mastery stayed at `1.0`, the
source slot was byte-stable, old-source replay during target adaptation was
zero, and controller updates were zero. Wrong-context, corrupted-target,
fresh-model, persistence, and frozen-controller controls were recorded.

This is the first promoted canonical result that transfers both capability
and measured learning speed through a richer external model state. The
reported speed gain is conditional: contexts are supplied, the task family is
small, target transition batches are reused and accounted for, and the bank is
append-only. It is not yet general continual learning. The next bottlenecks
are learned context formation, alternating-regime stress, and model-slot
consolidation/compression. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_transition_model_bank_continual_promoted_2026-08-09/`.

## Learned opaque transition context formation (2026-08-09)

The next two-seed rung replaces supplied context vectors with
`ExternalTransitionContextEncoder`. The encoder consumes only opaque state,
intention, next-state, and generic confidence tensors. It is trained outside
the controller with a paired noisy-view contrastive objective: the same
transition bundle should produce a stable key, while different bundles remain
separable. A held-out target dynamics bundle then grows a new
`ExternalTransitionModelBank` slot from the learned key; no regime label, task
ID, or privileged simulator field is passed to the learner.

Same-bundle clean/noisy cosine similarity was at least `0.9992` in both seeds,
and all context-formation, automatic-slot-growth, frozen-controller, prior
retention, wrong-context, corruption, fresh-model, byte-stability, and
persistence gates passed. The held-out target reached `1.0` measured mastery
after 21 and 19 optimizer updates, versus 38 and 26 for matched fresh models.
Both earlier factual slots retained `1.0` measured mastery; old prior examples
replayed during target adaptation were zero. The context encoder itself used
500 external optimizer updates and its training views are accounted for
separately.

This promotes learned context formation for a finite transition bundle. It
does not establish online identity formation over an alternating unbounded
stream: the encoder is trained before the held-out target is evaluated, the
model bank is append-only, current target examples are reused, and there is no
slot consolidation or compression. The next bottleneck is therefore online
context identity and capacity management under alternating regimes. Evidence
is archived in
`session_records/sequence_working_memory_2026-08-02/external_learned_transition_context_promoted_2026-08-09/`.

## Online transition-context identity under alternating regimes (2026-08-09)

The next two-seed rung adds `ExternalOnlineTransitionContextRouter`. Individual
opaque transitions are not assumed to identify a regime: some distinct
dynamics produce the same saturated next state. The router therefore buffers a
finite current-stream window, compares aggregate factual prediction errors and
a best-vs-second-best margin, and only then either routes the whole window to
an existing slot or admits a new opaque context. Ambiguous windows are not
written to any model.

The stream alternated base, auxiliary, base, held-out target, auxiliary,
target, and base regimes. Both seeds routed the mastered regimes to the
correct isolated slots, admitted the target exactly once without a regime
label, reused it on return, and reached `1.0` target mastery. Target adaptation
used 22 updates versus 31 and 37 for matched fresh models. Both prior slots
remained byte-stable with `1.0` retention; old-prior replay during target
learning was zero. A fourth regime at the three-slot capacity limit generated
a capacity result without growing or modifying the bank. Wrong-context MSE,
corruption, frozen-controller, and exact persistence controls passed.

This promotes bounded online identity and safe ambiguity handling, not
unrestricted continual learning. The encoder is pretrained, identity is
windowed rather than instantaneous, capacity pressure is refused rather than
solved by consolidation/compression, and current target windows are replayed
and accounted for separately. The next bottleneck is learned capacity
management: verified consolidation/compression must preserve all alternating
capabilities while allowing the external memory to grow beyond its fixed
window and slot budget. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_online_transition_context_promoted_2026-08-09/`.

## Verifier-gated transition-model capacity growth (2026-08-09)

The online router now gives the external model bank explicit bounded-capacity
state and a transactional `grow_verified` operation. Growth changes capacity
metadata only; the bank computes a content digest over all opaque keys and
model weights, runs a caller-owned held-out retention probe before and after
the transaction, and rolls back if content or retention changes. The router
and bank capacities must remain synchronized.

In the two-seed alternating pressure test, a fourth regime was first refused
at capacity three without writing or modifying the three existing slots. A
retention probe then passed for base, auxiliary, and target regimes, growth to
capacity four committed with an unchanged content digest, and the fourth
regime reached `1.0` measured mastery. Earlier regimes remained at `1.0`, the
controller stayed frozen, old-prior replay remained zero, and persistence,
wrong-context, corruption, and fresh-model controls passed.

This promotes safe capacity growth, not consolidation. Distinct transition
functions must not be merged just to reduce slot count; the next bottleneck
is a verifier-gated consolidation/compression candidate that preserves every
slot's held-out behavior, with rejection as the default when equivalence is
not demonstrated. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_transition_model_capacity_growth_promoted_2026-08-09/`.

## Behavior-verified transition-model consolidation (2026-08-09)

The bank now supports safe parameter-sharing consolidation. Two opaque context
keys remain present and addressable, but their model objects may share one
parameter set only when caller-supplied held-out transition predictions are
equivalent and an optional retention probe passes before and after the
transaction. The content digest records aliasing, and payload restore
reconstructs the sharing relationship.

Across two seeds, equivalent source/duplicate slots reduced physical model
objects from three to two while preserving all three context keys and factual
loss. A source slot and a disjoint target slot produced held-out prediction
differences of `0.174` and `0.126` and were rejected without mutation. The
controller and consolidation path performed zero optimizer updates; exact
persistence and wrong-context controls passed.

This promotes behavior-verified parameter sharing, not semantic merging or
unbounded compression. A finite held-out probe can miss an untested
difference, and the current mechanism does not yet learn low-rank/shared
representations for genuinely complementary functions. Evidence is archived
in
`session_records/sequence_working_memory_2026-08-02/external_transition_model_consolidation_promoted_2026-08-09/`.

## Held-out verified external transition-model compression (2026-08-09)

The bank now exposes compressed payload candidates using the existing external
storage codecs and restores them into independent runtime candidates. A
caller-owned retention probe must pass before a codec is promoted; candidate
construction performs no controller or model optimizer updates. Alias metadata
is preserved through compression and restore.

Across two seeds, float16 reduced model-state bytes from `26,944` to `13,472`
with held-out loss deltas below `4e-8`; int8 reduced them to `6,784` with
deltas below `9e-6`. Both codecs passed retention. Int4 reduced storage to
`5,176` bytes but was rejected by the same `1e-4` retention tolerance because
its loss drift was `2.7e-4`–`7.6e-4`. The controller and compression path used
zero optimizer updates, and compressed payload persistence passed.

This promotes external storage compression, not live reduced-precision
reasoning or new computation. Codec choice remains verifier-dependent and
finite held-out probes can miss rare failures; the next frontier is adaptive
codec selection and longer-horizon/alternating validation before compressed
artifacts are promoted. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_transition_model_compression_promoted_2026-08-09/`.

The codec boundary also exposes adaptive selection: it evaluates each caller
candidate with the same probe and selects the smallest accepted artifact. In
both compression seeds it selected int8 over float16 after rejecting int4.

## Accounted policy-free model compounding (2026-08-09)

The exported games session supplies the most important design correction for
continual learning: a policy stores preferences and can become wrong on a new
task, while a transition model stores factual dynamics and can be extended;
behavior can then be recomputed by search for the current opaque goal. The
games session also exposed two measurement hazards that are now normative here:
zero-shot capability is not the same claim as faster acquisition, and an
internal model-loss threshold is not deployed mastery.

`experiments/external_transition_model_compounding/` turns those lessons into
a fast canonical audit. One source regime is followed by three target regimes.
Each target is initialized from the preceding external transition model and
adapted in isolation; a matched fresh model is trained for every target. The
stopping prefix requires both transition loss and planner mastery. Reports
charge source acquisition in cumulative cost, keep current-target reuse
separate from old-regime replay, and record planner expansions and latency
separately from optimizer work.

Both seeds passed: warm target updates were `24/23/17` and `25/22/17`, versus
fresh `38/38/34` and `35/38/41`; cumulative warm cost ended at `1,264` versus
fresh `1,310` and `1,314`. Every target mastered, earlier models remained at
`1.0` and byte-stable, the controller stayed frozen, and old-regime replay was
zero. This promotes a replicated downward acquisition-cost signal in one
nested dynamics family. It does not establish general continual learning:
the fixture is small, context vectors are supplied, and the dynamics are not
yet genuinely disjoint across a wider family. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_transition_model_compounding_promoted_2026-08-09/`.

## Online routing over disjoint dynamics (2026-08-09)

The next pressure test removes the nested movement geometry. Four regimes
share one opaque state/intention interface but use unrelated transition tables.
`experiments/external_disjoint_dynamics_online/` feeds the online router one
row at a time. Rows remain provisional until the router's factual prediction
error and margin test fail, after which the frozen context encoder forms an
opaque key and a new external model slot is admitted. The router receives no
regime labels; labels in the experiment are diagnostics only.

All three seeds admitted two novel regimes once, reused each on a later visit,
and reached `1.0` planner mastery for all four regimes. Warm target updates
were `36/32` versus fresh `44/43` on seed `70411`, `35/31` versus `40/35` on
seed `70412`, and `27/30` versus `30/33` on seed `70413`. Source models stayed
at `1.0` and byte-stable, old-slot optimizer updates were zero, wrong-context
factual error passed, the controller stayed frozen, and persistence was exact
in every seed.

This promotes a stronger bounded claim than nested-delta transfer: factual
model routing and acquisition survive disjoint transition functions. It still
does not establish general continual learning. The context encoder is
pretrained on two source regimes, the evidence is a finite transition bundle,
and the planner has a finite horizon. The next frontier is online context
formation from partial multimodal event streams, model-bank growth under
unbounded alternation, and compression/consolidation validated over longer
horizons. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_disjoint_dynamics_online_promoted_2026-08-09/`.

## Long alternating disjoint-dynamics retention (2026-08-10)

The same four-regime fixture was then repeated four times in one stream. This
adds no model-training replay after acquisition; it tests whether factual
model selection remains stable as the active regime changes repeatedly. All
three seeds passed: each novel regime was admitted once and then correctly
routed seven times after admission, all four regimes retained `1.0` planner
mastery, source slots stayed byte-stable, old-slot optimizer updates stayed at
zero, and the controller remained frozen. Warm target fitting retained the
same advantage over fresh fitting as the shorter audit: `36/32` versus
`44/43`, `35/31` versus `40/35`, and `27/30` versus `30/33` across seeds
`70411`, `70412`, and `70413`.

This promotes longer-horizon routing stability for a bounded external factual
model bank. It does not establish noisy or partial multimodal context
discovery, unrestricted memory growth, learned consolidation/compression, or
general continual learning. Reports and the accounting ledger are archived in
`session_records/sequence_working_memory_2026-08-02/external_disjoint_dynamics_long_alternation_promoted_2026-08-10/`.

## Partial-evidence disjoint-dynamics retention (2026-08-10)

The next test withheld transition rows from the online router while preserving
only a verifier-private target-covering subset for each regime. Across three
seeds, `25` of `56` unique transition rows were observed and `31` were
withheld. The stream still repeated four alternating rounds. Every seed
admitted both novel regimes once, routed each seven times after admission, and
retained `1.0` planner mastery for all four regimes. Warm target fitting used
`20/20`, `14/15`, and `22/19` updates versus fresh `44/43`, `40/35`, and
`30/33`; source slots remained byte-stable, old-slot updates were zero, the
controller stayed frozen, and persistence was exact.

This promotes bounded partial-evidence routing, not arbitrary missingness: the
withheld rows were selected by the verifier-private fixture so the measured
planner targets remained solvable. It does not establish noisy multimodal
context discovery, unrestricted memory growth, learned consolidation or
compression, or general continual learning. Reports and accounting are in
`session_records/sequence_working_memory_2026-08-02/external_disjoint_dynamics_partial_evidence_promoted_2026-08-10/`.

## Noisy partial-evidence disjoint-dynamics retention (2026-08-10)

The partial-evidence fixture was then combined with Gaussian perturbations of
standard deviation `0.02` on every observed state and next-state tensor. Across
three seeds, `25` of `56` transition rows were observed and `31` withheld, with
four alternating rounds. Every seed admitted both novel regimes once, routed
each seven times after admission, and retained `1.0` planner mastery. Warm
target fitting used `20/20`, `14/15`, and `20/19` updates versus fresh
`44/43`, `40/35`, and `30/33`; source slots stayed byte-stable, old-slot
updates were zero, the controller stayed frozen, and persistence was exact.

This promotes one bounded noisy target-covering partial-evidence condition over
finite opaque transition tables. The mask is verifier-private and selected so
the measured targets remain solvable, and the noise is synthetic at one fixed
level. It does not establish arbitrary missingness, real multimodal noise,
unrestricted memory growth, learned consolidation or compression, or general
continual learning. Reports and accounting are in
`session_records/sequence_working_memory_2026-08-02/external_disjoint_dynamics_noisy_partial_promoted_2026-08-10/`.

## Rejected arbitrary missingness routing (2026-08-10)

Replacing the target-covering mask with a deterministic random half-mask
exposed the current identity bottleneck. Across seeds `70411`, `70412`, and
`70413`, the router saw `7/14` rows per regime window but could not reliably
reuse the same slot when the subset changed. Target-C reuse was `0/0/0`,
target-D reuse was `1/0/0`, and duplicate admissions exhausted capacity. Final
target mastery was respectively `0.333/1.000`, `0.333/0.333`, and
`0.000/1.000` for C/D.

This rejects the current router for arbitrary missingness. The failure is not
controller plasticity: the controller remained frozen, while incomplete
factual windows were mistaken for novel contexts. The next fix must create a
persistent sparse identity or evidence-accumulation boundary, with a
verifier-gated safeguard against contaminating an existing slot with a truly
novel regime. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_disjoint_dynamics_random_missingness_rejected_2026-08-10/`.

## Promoted sparse identity and compact consolidation (2026-08-10)

The rejected boundary was repaired with a compact slot-local evidence index.
Each slot stores unique factual `(state, intention, next-state)` records with
running means. A partial query may reuse a slot only when enough overlapping
facts agree and no overlapping fact contradicts it; unknown rows do not force a
duplicate. Sparse reuse is disabled while the bank is bootstrapping, preventing
shared facts from binding a novel regime to an old slot. Once the bank is
populated, each addressed factual model receives bounded consolidation updates
from its deduplicated external facts.

Across seeds `70411`, `70412`, and `70413`, random half-masks (`7/14` rows per
window) were alternated for four rounds. All three seeds admitted both novel
regimes once, routed each seven times after admission, retained `1.0` planner
mastery for every regime, kept source slots byte-stable, applied zero old-slot
updates, and passed exact persistence. The compact evidence index ended with
`56`, `56`, and `54` unique records. Warm target updates were `39/39`,
`38/39`, and `39/38`, versus fresh `44/43`, `40/35`, and `30/33`.

This is a qualified promotion of sparse identity, retention, and compact
consolidation. The model reused `632`, `615`, and `623` compact external fact
rows during sparse adaptation and consolidation; raw-row replay remained zero.
Uniform sample-efficiency improvement is explicitly not promoted: the latter
two seeds were slower than fresh controls. The mask is synthetic, bootstrap
admission is capacity-guarded, and this does not establish arbitrary real
multimodal missingness, unrestricted memory growth, or general continual
learning. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_disjoint_dynamics_sparse_identity_promoted_2026-08-10/`.

## Noisy sparse identity retention (2026-08-10)

The sparse identity boundary was repeated with Gaussian perturbations of
standard deviation `0.02` on observed state and next-state tensors. Random
half-masks remained in place (`7/14` rows per window) and the four-regime stream
still alternated for four rounds. All three seeds admitted both novel regimes
once, routed each seven times after admission, retained `1.0` planner mastery,
kept source slots byte-stable, applied zero old-slot updates, and passed exact
persistence. Compact external-fact reuse was `632`, `628`, and `623` rows;
raw-row replay remained zero.

This strengthens the qualified sparse-identity promotion to one synthetic
noise level. It does not establish real multimodal noise, unrestricted memory
growth, or general continual learning; the latter two seeds remain slower than
fresh controls. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_disjoint_dynamics_sparse_identity_noisy_promoted_2026-08-10/`.

## Copy-on-write transfer challenger for disjoint compounding (2026-08-09)

The exported session's next-step requirement was seed-widening on genuinely
different dynamics, while keeping the factual-model/policy distinction. The
new `ExternalTransitionModelBank.select_verified_transfer_prior()` API creates
isolated transfer and fresh candidates, lets a caller run a bounded
current-target factual probe, and returns an auditable selection receipt. The
live source slot is checked byte-for-byte before the caller explicitly appends
the selected candidate. This makes negative transfer measurable and
reversible without putting a policy or target label into the controller.

`experiments/external_transition_model_disjoint_compounding/` applies the
primitive to two source and two sequential target regimes with matched fresh
controls. Across five seeds, four warm runs had lower cumulative model-update
cost (`155/158`, `123/146`, `122/136`, and `146/155`) while retaining every
earlier slot at full planner mastery and byte stability. One seed was more
expensive (`141/139`), so the aggregate was correctly rejected rather than
promoted. A wider eight-update probe did not remove that failure, and a
sixteen-update probe was worse on all five seeds because challenger overhead
dominated.

The supported architectural result is the copy-on-write, verifier-gated
challenger—not reliable general compounding. The remaining bottleneck is
predicting full acquisition cost from a small factual prefix. The next useful
test is a cost-aware or multi-prefix challenger with a strict held-out gate,
followed by a wider disjoint-dynamics family. Evidence, including the
negative seed, is archived in
`session_records/sequence_working_memory_2026-08-02/external_transition_model_disjoint_compounding_rejected_2026-08-09/`.

## Variable-prefix online identity rejection (2026-08-09)

The next experiment trained `ExternalTransitionContextEncoder` with a new
prefix-alignment objective and attempted admission after only `7` of `14`
transition rows. The online router now maintains an opaque active-slot
continuation state and conflict patience, so an undertrained current model no
longer immediately mints a duplicate slot. Protected source slots are
read-only during target adaptation.

The identity portion worked: both seeds formed short-prefix admissions, later
reused the target-C slot, and preserved source slots at `1.0` with unchanged
digests. The capability portion failed: target-D reached only `0.333` mastery
on both seeds, target-C reached `0.0`/`1.0`, and online updates were not
consistently cheaper than fresh models. This is rejected evidence, not a
promotion. The remaining bottleneck is credit assignment and evidence
accumulation from partial streams—how to turn provisional observations into a
useful model without either polluting a protected slot or minting duplicates.
Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_partial_evidence_identity_rejected_2026-08-09/`.

## Copy-on-write provisional model candidates (2026-08-09)

The router now supports `defer_admission=True`: a novel context creates a
provisional `ExternalTransitionModel` outside the committed bank. Caller-owned
updates affect only that candidate. The candidate retains a cumulative opaque
evidence window across staged observations, so later updates can assign credit
using all verified current-stream evidence without touching protected slots.
`promote_staged_candidate` clones the bank, checks held-out transition error,
runs a caller-owned retention probe, and only then commits the candidate as a
new slot. A failed proof leaves the committed bank unchanged.

The initial sparse-evidence audit was intentionally rejected: copy-on-write
isolation passed, but four current rows did not generalize reliably to two
held-out rows. After adding cumulative candidate evidence-window training, the
two-seed rerun passed held-out prediction with errors `0.129` and `0.143` at
tolerance `0.2`; source slots remained byte-stable, the frozen controller was
unchanged, and exact payload persistence passed. Four unique target rows were
presented `600` times to the candidate (`596` candidate-evidence replayed
rows), while old committed-slot replay stayed zero. This promotes a bounded
credit-assignment mechanism, not replay-free general continual learning or
unrestricted memory growth. Evidence from the rejection remains archived in
`session_records/sequence_working_memory_2026-08-02/external_provisional_candidate_promotion_rejected_2026-08-09/`.
The promoted rerun is archived in
`session_records/sequence_working_memory_2026-08-02/external_provisional_candidate_promotion_promoted_2026-08-09/`.

A strict one-pass control then disabled candidate-window replay. Both seeds
preserved controller/source isolation and exact persistence, but held-out
errors rose to `0.882` and `0.778` against tolerance `0.2`; no older candidate
evidence was replayed. This rejects the claim that the current fixed MLP can
learn the sparse target stream in one pass. Repeating the current bundle was
accounted separately from old-evidence replay. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_provisional_candidate_one_pass_rejected_2026-08-09/`.

## Replay-free affine sufficient-statistics memory (2026-08-09)

The rejected one-pass MLP control identified a missing class of mechanism:
some transition structure can be learned exactly from compact sufficient
statistics, without retaining or replaying raw observations. The new
`ExternalAffineTransitionStatistics` component accumulates weighted normal and
target matrices over opaque state/intention/next-state tensors and persists
only those matrices plus a sample count.

Across two seeds, `12` training rows were consumed once and `4` held-out rows
were predicted with errors `1.84e-13` and `2.58e-14`. Both raw-evidence absence
and exact persistence passed, with zero optimizer updates and zero replayed
examples. This promotes a narrow replay-free affine memory primitive, not a
general nonlinear learner. Its next architectural use is as a replaceable
fast path or expandable component alongside the general external transition
model, with verifier-gated selection rather than hand-assigned semantics.
Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_affine_one_pass_promoted_2026-08-09/`.

The primitive is now available through the same `ExternalTransitionModelBank`
boundary as the nonlinear MLP. A bank can create uncommitted candidates from
either family, route affine updates directly into sufficient statistics without
an optimizer, and round-trip either family through the bank checkpoint. The
bank also exposes verifier-gated family selection: independently adapted
candidates are compared on held-out factual prediction, optional retention,
and storage size, then the smallest accepted candidate is reported without
mutating the live bank. This removes a hand-assigned “affine task” decision,
but it does not yet train both candidates automatically or support mixed model
families within one committed bank.

## Alternating provisional-candidate isolation (2026-08-09)

The previous candidate boundary had one mutable provisional model. If a
second novel stream arrived before the first candidate was promoted, its
observations could be folded into the first evidence window. The router now
keeps a bounded list of isolated provisional candidates. Factual candidate
prediction selects the matching candidate for continuation; a conflicting
stream stages a new candidate when capacity permits. Adaptation and promotion
carry the candidate index, and the full candidate list—including models and
evidence windows—persists through the payload boundary.

The focused alternating-stream regression passed: two novel candidates were
staged without a committed-bank write, payload restore preserved both, the
second candidate was promoted through the verifier and source-retention
gates, and the first candidate's digest remained unchanged. The complete suite passed
`408` tests. This is an interference-safety and bounded-quarantine result, not
yet general continual learning: candidate capacity is finite, candidate
evidence is still replayed within each quarantine, and multi-candidate
promotion still requires caller-owned held-out verification.

## Candidate-driven verifier-gated capacity growth (2026-08-09)

The candidate lifecycle now closes the full-capacity transition. When a
committed bank is full, `promote_staged_candidate` may receive a larger
`destination_capacity`. The disposable candidate bank expands capacity as
metadata only, checks that its content digest is unchanged, appends the
candidate, and runs the held-out and caller-owned retention gates. Only after
both pass does the live bank capacity and slot change; a rejection leaves both
unchanged. The router's logical candidate capacity is updated atomically with
the bank.

A focused regression passed this transaction with a one-slot bank: the
candidate was promoted into capacity two, the source slot remained byte-stable,
and the router and bank capacities stayed synchronized. This improves the
memory lifecycle but remains bounded continual learning; it does not provide
unrestricted growth, learned eviction, or compression of active candidate
evidence.

## Verifier-gated external model-slot eviction (2026-08-09)

The model bank now supports `evict_verified(index, retention_probe)`. It
constructs a disposable payload clone, runs the caller's retention probe on
the live bank and again after removing one logical context, then commits the
new context/model lists only when both pass. Payload reconstruction preserves
shared model aliases, so removing one aliased tail cannot delete another
context's parameters. Middle-slot eviction is rejected because it would
renumber opaque addresses; stable slot IDs or an explicit remapping protocol
are still future work. Failed probes leave the live bank and digest unchanged.

The focused regression accepted removal of an aliased tail while retaining the
source and another context, then rejected a non-tail/destructive probe without
mutation. This closes safe capacity reuse, but it is not learned
eviction: the retention probe remains caller-owned, and no slot is removed
without an explicit verifier proof.

The router-level lifecycle regression also passed a complete grow → promote →
tail-evict → stage → promote cycle. Eviction invalidates a stale active-slot
reference before the next candidate is learned, and the original source model
remains byte-stable throughout. This is the current safe route to reuse finite
capacity while stable logical addresses remain an open design requirement.

## Mixed-family external slots and automatic candidate selection (2026-08-09)

The external bank now supports a verifier-selected mixed mode. Each committed
slot carries its own opaque implementation family, so an affine sufficient-
statistics slot and a nonlinear MLP slot can coexist, transfer only within the
same family, persist, alias, compress, and evict without changing the
controller boundary. A mixed router quarantines both families for a novel
context and adapts them independently: sufficient statistics consume the
current verified observation once, while nonlinear candidates use caller-owned
optimizer state.

The nonlinear path also has a bank-owned SGD fallback with an explicit learning
rate in the external configuration. Callers may still inject a richer
optimizer, but the controller and runtime no longer need to construct one just
to perform an isolated external update. This removes an interface dependency,
not the learning difficulty: the fallback is still a gradient update and does
not make the nonlinear model replay-free or one-pass competent.

Promotion evaluates all adapted candidates on held-out factual evidence, then
commits only the smallest accepted family and runs the existing bank retention
probe. The focused regression also round-trips both provisional candidates
before promotion. This removes the remaining hand-assigned family choice from
the promotion path, but it is still a bounded model-family set: new families,
automatic optimizer provisioning, and replay-free nonlinear adaptation remain
open continual-learning bottlenecks for the general nonlinear model.

## Replay-free fixed nonlinear features (2026-08-09)

The affine primitive's one-pass limitation is now pressure-tested with a fixed
cosine feature map. `ExternalRandomFeatureTransitionStatistics` persists the
feature projection and updates only weighted normal/target matrices, so it
consumes nonlinear transition evidence once without retaining raw rows or
requiring optimizer state. Across two seeds, `64` nonlinear training rows were
presented once and `64` held-out rows were evaluated with errors below `0.02`;
exact payload restoration passed.

This family is available to the same mixed bank and router and is selected only
through held-out verifier evidence. It expands replay-free learning beyond
affine maps. Its basis can now grow through a retention-verified algebraic
transaction: old sufficient statistics are remapped and new features begin
with zero historical evidence, so no old rows are replayed. This is still not
unrestricted neural computation; distribution shift, repeated growth, and
long-horizon retention remain unverified before treating it as general
continual learning.

## Composed goal-conditioned lifecycle rewrites (2026-08-09)

The external lifecycle now has a two-seed composition audit covering
query-conditioned verifier-gated eviction, held-out verified consolidation of
equivalent survivors, and verified float16 compression. Both seeds preserved
retained factual behavior and stable logical addresses while preserving
bank-owned telemetry across normal and compressed checkpoints. Consolidation
reduced three physical models to one shared parameter object, and compression
selected `torch.float16`; controller updates and replayed transition examples
were zero. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_transition_lifecycle_rewrite_promoted_2026-08-09/`.

This closes bounded lifecycle composition. The next unresolved boundary is
long-horizon query-conditioned relevance across actual controller replacement
and multimodal representation drift; without that, the system still has a
safe bounded memory manager rather than general continual learning.

## Goal-conditioned external relevance (2026-08-09)

The ambiguity control identifies the missing signal; the corresponding
external boundary now consumes a learned opaque query aligned with the current
goal and compares it with stable memory context keys. Query/key cosine
alignment adjusts the lifetime proposal away from relevant slots, while the
generic telemetry scorer and verifier-gated copy-on-write transaction remain
independent and replaceable.

On two seeds, the query-conditioned transaction achieved `1.000` held-out
selection versus `0.500` for the matched-telemetry random reference. Retained
factual models passed held-out probes, policy checkpoints restored exactly,
and controller updates and replayed transition examples were zero. Evidence
is archived in
`session_records/sequence_working_memory_2026-08-02/external_transition_goal_conditioned_relevance_promoted_2026-08-09/`.

This promotes the retrieval boundary, not upstream query learning: the query
is treated as a learned output of the replaceable controller/event system.
The next bottleneck is verifying that learned queries remain stable across
modalities and controller replacements, then integrating verified
consolidation/compression into the same goal-conditioned lifecycle.

## Matched-telemetry future relevance control (2026-08-09)

The next control equalized bank-owned usage, age, and factual prediction-error
telemetry across four real transition models. Two candidate memories were
otherwise indistinguishable, while an opaque future schedule randomly selected
which one would be needed later. The lifetime policy achieved held-out
selection `0.460`/`0.595` across two seeds against a `0.500` random ceiling.

This result is decisively rejected as a general relevance mechanism. It passed
exact policy persistence, zero-replay, and zero-controller-update gates, but it
shows that more generic lifetime heuristics cannot infer information that is
absent from the current evidence. The next boundary must be a
goal-conditioned external query/relevance mechanism: learned current event,
intention, or goal state selects factual memory candidates, while the verifier
continues to authorize retention transactions. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_transition_future_relevance_ambiguity_rejected_2026-08-09/`.
The two-seed reports and accounting ledger are archived in
`session_records/sequence_working_memory_2026-08-02/external_random_feature_one_pass_promoted_2026-08-09/`.

## Long-horizon replay-free nonlinear slot retention (2026-08-09)

The retention pressure test learned four disjoint nonlinear transition
families into isolated sufficient-statistics slots, then revisited all four
after later slots had been acquired. With ridge `1e-4`, both seeds passed every
held-out floor, zero replay, unchanged slot digests, and exact bank
persistence. The rejected ridge `1e-5` run retained earlier slots perfectly
but failed the capability floor on two regimes; it is archived separately as a
controlled negative result.

This promotes bounded replay-free nonlinear retention with verifier-supplied
context keys. It does not yet demonstrate learned context formation,
unrestricted memory growth, or general continual learning. Evidence is
archived in
`session_records/sequence_working_memory_2026-08-02/external_random_feature_retention_promoted_2026-08-09/` and the rejection is in
`session_records/sequence_working_memory_2026-08-02/external_random_feature_retention_rejected_2026-08-09/`.

## Online learned-context nonlinear retention (2026-08-09)

The online router now separates committed-slot continuation tolerance from
provisional-candidate continuation tolerance. A new regime can therefore
reject the active committed model while allowing its quarantined statistical
candidate to accumulate evidence. Replay-free sufficient-statistics slots no
longer copy prior-regime statistics as transfer initialization; only trainable
nonlinear weights inherit a prior.

Across seeds `1601` and `1602`, four disjoint nonlinear streams arrived one row
at a time. The context encoder formed opaque candidate keys without regime
labels, all four promotions passed the `0.02` held-out floor, prior-slot
retention probes passed at every promotion, zero replay was recorded, and exact
router persistence passed. This is the strongest current retention result, but
the context encoder was not trained in this fixture and the memory capacity is
four slots. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_random_feature_online_retention_promoted_2026-08-09/`.

## Stable logical addresses under verified memory reorganization (2026-08-09)

The exported session's model-versus-policy result also implies a memory-system
requirement: a growing factual store must be able to reorganize without
silently changing the meaning of a persisted reference. `ExternalTransitionModelBank`
now assigns monotonic opaque slot IDs in addition to its backward-compatible
physical indices. IDs survive payload and compressed-payload round trips,
middle-slot removal, alias-preserving consolidation, and later capacity reuse;
the router resolves its cached active slot by logical ID after reorganization.

`evict_verified_id` performs the same copy-on-write, pre/post retention proof as
tail eviction but permits a verified middle removal. A focused regression
removed logical slot `1` from `(0, 1, 2)`, retained `(0, 2)` without renumbering,
reused a fresh ID `3`, and repaired the router's physical cache. Legacy payloads
without IDs remain readable through a compatibility checksum path. The full
suite passed `424` tests.

This closes address stability for the current bounded model bank. It does not
make eviction learned: retention probes, lifetime policies, and reclamation
decisions remain caller/verifier-owned. The next bottleneck is a learned,
auditable memory policy that can choose what to consolidate, compress, or evict
under long alternating streams while preserving the stable logical address
contract.

## Verifier-trained external lifetime proposals (2026-08-09)

The bounded bank now exposes `ExternalTransitionModelLifetimePolicy`. It is an
independently replaceable, permutation-equivariant scorer over opaque context
keys and generic usage, age, and factual-error telemetry. It proposes a stable
logical slot ID; protected slots are hard-masked before selection. The policy
cannot mutate the bank directly: `evict_verified_id` still performs the
copy-on-write pre/post retention proof, and the policy consumes only the single
verifier outcome bit from that transaction as an external update.

Focused controls pass permutation-equivalent scoring, all-protected refusal,
rejection without bank mutation, stable-ID middle eviction, exact policy
checkpoint restoration, and router active-reference repair. The full suite
passes `430` tests; the policy has no controller parameters or controller
optimizer updates.

The two-seed online audit then trained the policy from one verifier bit per
transaction and compared it with matched random and recency selectors. Learned
held-out selection was `0.780`/`0.820`, versus random `0.515`/`0.530` and
recency `0.265`/`0.295`; protected-slot, stable-address, persistence, and
zero-replay gates passed on both seeds. Evidence and accounting are archived
in
`session_records/sequence_working_memory_2026-08-02/external_transition_lifetime_policy_promoted_2026-08-09/`.

This promotes a bounded verifier-trained lifetime proposal mechanism, not
unrestricted learned eviction or general continual memory: the verifier
remains authoritative, the policy has not yet learned consolidation/compression
tradeoffs, and it has not survived a long alternating capability stream.

## Bank-owned lifetime telemetry under capacity pressure (2026-08-09)

The bank now owns generic lifetime telemetry keyed by stable logical slot ID:
access usage, logical age, and an exponentially smoothed factual
prediction-error signal. The telemetry is updated at the external transition
observation boundary, persists in normal and compressed bank checkpoints, and
is removed with the corresponding logical slot during verified copy-on-write
eviction. A lifetime policy can consume the bank directly through
`evict_from_bank_verified`, so callers do not need to manufacture feature
tensors or maintain a parallel physical-index ledger.

The two-seed capacity-pressure audit trained on one verifier bit per
transaction using three independently trained affine factual models per fresh
bank. Learned held-out safe selection was `0.575`/`0.555`, versus random
`0.340`/`0.280` and recency `0.015`/`0.020`. Retained-model held-out behavior,
protected-slot, stable-address, exact persistence, zero-replay, and
zero-controller-update gates passed. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_transition_lifetime_capacity_pressure_promoted_2026-08-09/`.

This is a real integration gain: the lifetime policy now operates on external
memory state produced by actual model use rather than fixture-supplied
telemetry. It remains bounded and verifier-dependent, not unrestricted
continual learning. The next bottleneck is a longer alternating stream with
genuinely retained versus disposable capabilities, where lifetime selection
must coexist with verified model growth, consolidation, and compression under
the same capacity budget.

## Long alternating retention under shared capacity (2026-08-09)

The next pressure test ran three recurring factual transition models through a
bounded four-slot bank while replacing a disposable fourth model across `600`
training and `240` held-out pressure events per seed. Every eviction proposal
was checked against held-out behavior for all recurring models. Both seeds
preserved every recurring capability at every measured prefix, with learned
safe admission `1.000`/`1.000`, versus random `0.500` and recency `1.000`.

This is a meaningful retention result, not a policy-superiority result:
recency is equally strong because the fixture intentionally makes disposable
models stale. Seed 1702 had five training misses before converging; the
verifier-authorized fallback preserved the recurring models after each miss.
Evidence and accounting are archived in
`session_records/sequence_working_memory_2026-08-02/external_transition_lifetime_capacity_stream_promoted_2026-08-09/`.

The remaining high-ROI bottleneck is sharper: make retained versus disposable
capabilities genuinely ambiguous to recency, then integrate verified
consolidation and compression decisions into the same lifecycle ledger. Until
that passes, the system has bounded retention with safe fallback, not general
continual learning.

## Goal-conditioned factual model search (2026-08-09)

The external planner now ranks runtime-variable factual models by predicted
goal reachability and returns a stable logical model address. In a two-seed
audit with three independently learned dynamics, selection reached `1.000`
versus `0.333` random, with a positive held-out goal margin on every
evaluation. No task policy was stored; controller updates and transition
replay were zero. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_transition_goal_model_selection_promoted_2026-08-09/`.

This is the strongest current route toward reusable capability: memory stores
facts and inference derives behavior from the current goal. The next pressure
test must vary the query representation and modality/frontend while keeping
the factual bank fixed, then verify that selected logical addresses and
retention floors survive that replacement.

## Explicit representation-space compatibility and migration (2026-08-09)

The external factual bank and planner now carry explicit state and intention
representation-space IDs. Goal-conditioned model search rejects equal-width
but semantically different replacements instead of silently interpreting them
as compatible. The IDs are persisted in bank configuration and included in its
digest; legacy payloads without IDs load into the documented `opaque-state-v1`
and `opaque-intention-v1` defaults.

Replacement is copy-on-write. A candidate bank can be approved only when its
stable logical addresses and opaque context keys match, held-out transition
predictions remain within tolerance, and an optional retention probe passes.
The live bank is not mutated by approval. This creates the needed interface
version gate without pretending that metadata relabeling repairs arbitrary
representation drift: a real controller/frontend replacement still needs a
learned alignment or a behavior-preserving migration candidate.

In the two-seed migration audit, the unchanged candidate was accepted with
zero held-out prediction difference, both drifted candidates were rejected,
and the old planner/bank mismatch was rejected before search. Evidence is
archived in
`session_records/sequence_working_memory_2026-08-02/external_transition_representation_migration_promoted_2026-08-09/`.

## Runtime-wide representation replacement (2026-08-09)

The same contract now reaches the canonical `N encoders -> event bus -> one
controller -> intention bus -> M decoders` runtime. The runtime records event,
controller-state, and intention space IDs, and exposes a copy-on-write
migration probe over paired source/target event windows. It compares the
controller's intention, execution decision, and continuation state while
leaving external memory out of the probe so memory migration remains an
independent contract.

Across two seeds and 24 held-out two-stream windows, a behavior-preserving
replacement passed with zero differences in all three measured outputs. A
candidate with changed controller behavior failed in both seeds. This is the
first runtime-wide compatibility gate, but it is still not learned alignment:
the target event representation is supplied by the caller and must already be
behavior-preserving. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/runtime_representation_migration_promoted_2026-08-09/`.

## Retention-safe external memory representation migration (2026-08-09)

Content-addressed memory now carries explicit opaque key and value space IDs.
Its copy-on-write migration gate requires a one-to-one mapping for every
occupied address, preservation of protected retention evidence, and held-out
query equivalence. Retention histories can be transferred to transformed keys
as state; verifier outcomes are not replayed. A caller-supplied value-space
alignment is allowed, but it remains an independently verified candidate.

Across two seeds, a key-permuted replacement preserved both held-out reads and
the protected row with zero value difference. A candidate with a changed
stored value was rejected in both seeds. This closes the memory migration
transaction boundary, but does not establish arbitrary learned value
alignment, unbounded storage, or general continual learning. Evidence is
archived in
`session_records/sequence_working_memory_2026-08-02/memory_representation_migration_promoted_2026-08-09/`.

## Outcome-only event-space alignment under frozen computation (2026-08-09)

The first replicated learned-alignment rung now trains only a replaceable
external event bridge after freezing the parent controller, external register,
and output decoder. The trainer cyclically permutes the learned event tensor,
masks controller state, and supplies the bridge only the sampled scalar
verifier outcome from the opaque action record. A matched reward-shuffled arm
and digest checks prevent the result from being explained by privileged
representation labels or parent drift.

Across seeds `69316` and `69317`, source capability was `0.941`/`0.992`, the
changed event space was `0.516`/`0.727` before bridge adaptation, and the
outcome-trained bridge reached `0.992`/`0.996`. Shuffled-outcome controls stayed
at `0.559`/`0.613`; source retention and every frozen-component digest gate
passed. Stable prefixes appeared at `4,096` and `2,048` unique verifier bits,
with zero replayed examples. The full reports and accounting ledger are in
`session_records/sequence_working_memory_2026-08-02/outcome_only_event_alignment_promoted_2026-08-09/`.

This is a meaningful foundation result: a frozen computational substrate can
learn a bounded event-space correction through an isolated plastic boundary.
It is not yet arbitrary multimodal alignment or general continual learning.
The next bottleneck is to replace the known cyclic permutation with unknown,
composed, and eventually modality-specific representation drift while keeping
the same scalar-only credit and retention controls.

## Outcome-only composed event alignment (2026-08-09)

The alignment pressure test now removes the coordinatewise shortcut. A fixed,
opaque dense orthogonal transform mixes all 32 event features before the
replaceable bridge. The bridge receives no transform description and no
controller state; it is trained only from sampled scalar outcomes on opaque
actions. A matched reward-shuffled arm uses the same transform, architecture,
and verifier budget.

Across seeds `69316` and `69317`, source capability was `0.941`/`0.992`,
transformed capability was `0.500`/`0.504` before adaptation, and the bridge
reached `0.949`/`0.984`. Shuffled-outcome controls stayed at `0.445`/`0.594`;
source retention, frozen parent, external register, and decoder digest gates
all passed. Stable prefixes appeared at `10,240` and `6,144` unique verifier
bits, with zero replayed examples. Reports and accounting are archived in
`session_records/sequence_working_memory_2026-08-02/outcome_only_composed_event_alignment_promoted_2026-08-09/`.

This materially strengthens the representation result: outcome-only bridge
learning can recover a dense composed invertible change, not only a known
cyclic coordinate shift. The remaining gap is unknown transform identity and
composition across changing frontends or modalities, followed by
non-invertible information loss and long-horizon continual retention.

## Persistent external alignment-cell stream (2026-08-09)

Alignment is now pressure-tested as growable external state. For each of
three opaque dense frontend transforms, the stream allocates a fresh
`neural-computer.event-bridge.v1` cell, trains it from scalar verifier
outcomes, freezes it on admission, and then returns to all earlier cells
without replaying their acquisition examples. The parent controller, source
register, and decoder remain frozen throughout.

Both seeds passed the full stream gates. Every cell mastered; every prior cell
retained mastery after later admissions; reward-shuffled controls failed for
all cells; and zeroing one cell degraded only that cell. Seed `69316` ended at
`0.957`/`1.000`/`1.000`, while seed `69317` ended at `0.984`/`0.980`/`0.992`.
Stable prefixes were `10,240`/`4,096`/`6,144` and `6,144`/`4,096`/`4,096`
verifier bits, with zero replayed examples. Reports and accounting are
archived in
`session_records/sequence_working_memory_2026-08-02/outcome_only_alignment_cell_stream_promoted_2026-08-09/`.

This is the first bounded no-replay growth result at the alignment boundary:
external plastic state can expand while retaining earlier learned interfaces.
The decisive remaining bottleneck is automatic addressing: the system still
needs to infer which opaque alignment cell applies to the current stream,
without a task or frontend identity, and must do so under longer sequences,
cell eviction, and genuinely ambiguous drift.

## Outcome-only automatic alignment-cell addressing (2026-08-09)

The external alignment bank now has a learned router. It receives only pooled
statistics of the current learned event tensors, selects an opaque cell, and
learns from the scalar verifier outcome of the selected action. Transform
seeds, frontend identities, task labels, and correct actions are excluded from
the router optimizer. A reward-misaligned control delivers the previous
episode's scalar outcome to the current choice.

Across seeds `69316` and `69317`, routing accuracy was `1.000` in both runs.
Selected-cell action accuracy was `0.965`/`0.988`/`0.992` and
`0.984`/`0.977`/`0.996`; shuffled routers reached only `0.333` and `0.667`
routing accuracy and failed action mastery. The three-cell growth,
no-replay retention, single-cell corruption, and frozen-core gates all passed.
Reports and accounting are archived in
`session_records/sequence_working_memory_2026-08-02/outcome_only_alignment_cell_routing_promoted_2026-08-09/`.

This closes the bounded stream's addressing gap: external alignment cells can
grow, retain older capabilities, and be selected from learned event evidence
using scalar credit. The next frontier is more demanding: unknown frontend
changes must be discovered online rather than drawn from a fixed registered
transform set, and the bank must support eviction/consolidation while keeping
the same no-replay retention guarantees.

## Outcome-only online admission with immutable alignment keys (2026-08-09)

The alignment bank now admits an unregistered fourth dense event transform
after three cells have already been mastered and routed. The new bridge learns
from scalar verifier outcomes, while old bridge cells and old address keys are
frozen. During the growth phase, no old-stream examples are replayed. The new
cell's address is an opaque event-signature key appended to external memory;
it is not a task or frontend identity.

Across seeds `69316` and `69317`, the new cell reached `0.984`/`0.980`, its
shuffled controls reached `0.496`/`0.449`, and immutable-key routing selected all
four cells correctly with action mastery. Key corruption dropped routing to
`0.5` in both seeds. A matched shared-head expansion failed old-route
retention in both seeds, even with old output rows frozen; this is retained as
the negative control that motivated immutable append-only keys. Reports and
accounting are archived in
`session_records/sequence_working_memory_2026-08-02/outcome_only_online_alignment_growth_promoted_2026-08-09/`.

This closes bounded online admission and no-replay address retention. The
remaining frontier is now lifecycle management: the key bank must handle
ambiguous or drifting signatures, capacity pressure, eviction, consolidation,
and recovery after corrupted or stale cells without replaying old experiences.

## Outcome-only reversal-safe alignment-cell lifecycle (2026-08-09)

The external alignment bank now has a capacity-pressure transaction. Four
cells are first mastered and protected by the scalar retention ledger. One
cell is then evaluated under a changed event space; four low verifier
observations release it, the ledger selects only that stale slot for eviction,
and a newly trained replacement is admitted under the same logical address.
The three protected cells and frozen source computation are retained.

Both seeds passed the full lifecycle gates: base mastery and shuffled nulls,
all-base protection, scalar reversal, exact stale-slot eviction, replacement
mastery and routing, protected-cell retention, and frozen-core digests. The
reversal threshold is `0.75` against a mastery threshold of `0.8`; zero
replayed examples were used. Reports and accounting are archived in
`session_records/sequence_working_memory_2026-08-02/outcome_only_alignment_lifecycle_pressure_promoted_2026-08-09/`.

An earlier `0.70` reversal threshold safely refused one seed because verifier
noise produced stale outcomes of `0.703` and `0.734`. That negative result is
important: reversal hysteresis must be calibrated against outcome noise, and
the lifecycle must refuse eviction when evidence is ambiguous. The next
frontier is learned consolidation/compression and genuinely ambiguous
capacity pressure, not simply adding more cells.

## Raw-evidence-free streaming transition candidates (2026-08-09)

The online factual-memory boundary now has an explicit
`provisional_evidence_policy`. Its `streaming_statistics` mode is restricted to
model families that expose a one-pass `observe` operation, such as the fixed
random-feature sufficient-statistics model. Each provisional transition row is
consumed once into external statistics; the candidate persists only its opaque
context key, model statistics, and an evidence count. Raw candidate rows are
not retained and cannot be replayed accidentally during adaptation or payload
restore. The new `streaming_gradient` mode gives caller-owned learned models the
same raw-evidence boundary: current windows are optimized without retaining
provisional rows, while local repeated updates are accounted separately from
old-regime replay. The existing cumulative-window mode remains available for
general learned MLP candidates, whose replay requirement is now explicit rather
than hidden behind a nominal “online” path.

Across seeds `1801` and `1802`, three nonlinear transition regimes were
staged and promoted from `64` rows each. Held-out factual errors were
`[0.003073, 0.000021, 0.002236]` and `[0.001973, 0.000053, 0.013429]`, all
below the `0.02` promotion threshold. Shuffled-next-state controls reached
`1.646` and `0.915` and were rejected. Every prior slot retained its held-out
behavior, the controller remained byte-stable, persistence was exact, and
zero replayed examples or raw provisional rows were recorded. Evidence is
archived in
`session_records/sequence_working_memory_2026-08-02/external_streaming_statistics_candidate_promoted_2026-08-09/`.

This promotes a raw-evidence-free provisional boundary for bounded nonlinear
sufficient-statistics models. It does not make arbitrary nonlinear MLP
learning one-pass or replay-free, and it does not establish unrestricted
continual learning. The next bottleneck is to build a similarly explicit
streaming learner for genuinely changing, partial dynamics rather than
silently retaining a replay window.

## Interleaved streaming factual candidates (2026-08-09)

The streaming-statistics boundary now has an interleaved pressure test. Two
novel dynamics streams arrive in alternating four-row windows before either
candidate is promoted. The router keeps separate copy-on-write candidates,
updates each only through one-pass sufficient statistics, and offers both
affine and fixed random-feature families. The family is selected only by the
held-out factual promotion gate; no task or regime label is given to the
router. A provisional match margin now produces an explicit `ambiguous`
decision when two candidates explain a window too similarly; neither model is
updated and the window is not silently assigned.

Across seeds `1901` and `1902`, both candidates consumed `64` rows, retained
zero raw provisional rows, and were promoted as affine models. Held-out errors
were below `1e-6` in every source and target slot; shuffled-next-state controls
reached `37.49` and `13.99` and were rejected. A capacity-limited control
refused the second unverified stream without modifying the committed source
slot. Source retention, frozen-controller, exact-persistence, and zero-replay
gates passed. A deliberately ambiguous window was refused without a candidate
update in both seeds. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_interleaved_streaming_candidates_promoted_2026-08-09/`.

This closes a bounded interleaved-candidate isolation gap. It also sharpens
the next weakness: fixed random-feature candidates are not reliably
identifiable from very short nonlinear prefixes, whereas affine sufficient
statistics are. The system must learn a stronger streaming nonlinear model or
an evidence-accumulation policy that preserves ambiguity without mixing
streams; increasing capacity or silently replaying provisional rows is not an
acceptable substitute.

## Transactional quarantine for ambiguous streaming evidence (2026-08-09)

The interleaved streaming boundary now has an explicit ambiguity policy. With
`ambiguous_evidence_policy="quarantine"`, a bundle whose best provisional
candidate is within tolerance but below the required prediction-error margin is
copied into a bounded external quarantine. It changes no candidate statistics,
cannot be promoted while unresolved, is serialized with the router, and is
assigned only after a later observation produces a verified margin. The next
adaptation step consumes the deferred bundle once and clears it. Capacity
overflow refuses the bundle rather than silently mixing streams.

Across seeds `1901` and `1902`, the same two interleaved candidates first
consumed `64` rows each. A deliberately over-conservative margin quarantined
`4` rows, persisted and restored them, then resolved them to the correct target
candidate; post-resolution evidence counts were `[72, 64]` and the quarantine
was empty. Candidate raw-row retention remained `[0, 0]`, shuffled controls
were rejected at `37.49` and `13.99`, all held-out/retention/capacity/frozen
controller gates passed, and replayed examples remained `0`. The promoted
records are in
`session_records/sequence_working_memory_2026-08-02/external_interleaved_streaming_quarantine_promoted_2026-08-09/`.

This is a stronger evidence-preservation boundary, not a claim of general
continual learning: the quarantined rows are bounded external state and are
consumed once, not magically reconstructed. The next high-value test is to
replace the hand-set margin stress with a learned/calibrated reliability and
delay policy on partial nonlinear streams, while measuring positive transfer
against a fresh learner. The exported prior session reinforces the same
direction: a shared trajectory-statistics route is a promising reusable
mechanism, but safe storage alone did not produce positive transfer.

## Calibrated external reliability at streaming admission (2026-08-09)

The streaming router now accepts an optional independently versioned evidence
evaluator and threshold. For each candidate model, factual prediction error is
first computed as usual; when the candidate has crossed an explicit warm-up
evidence count, the evaluator may veto a low-error match. The warm-up is
important: applying a learned reliability gate to an untrained provisional
model starves it of the evidence needed to become useful. Before warm-up,
candidate learning remains isolated and bounded; after warm-up, calibrated
reliability controls continuation and ambiguity routing. Payload restore
requires the external evaluator explicitly, preserving component replacement
instead of reconstructing hidden dependencies.

Across seeds `2001` and `2002`, two nonlinear candidates consumed `64` rows
each through one-pass random-feature statistics and retained zero raw rows. A
corrupted four-row bundle had raw errors `0.00791` and `0.01138`, both below
the router continuation tolerance of `0.02`, but calibrated clean/noisy
probabilities were `0.989`/`0.248` and `0.906`/`0.179`; both corrupted streams
were rejected without changing candidate statistics. Held-out errors stayed
below `0.0045`, contextual calibration and router persistence passed, and the
controller digest remained unchanged. The promoted records are in
`session_records/sequence_working_memory_2026-08-02/external_calibrated_streaming_admission_promoted_2026-08-09/`.

This promotes bounded learned reliability at nonlinear streaming admission.
The evaluator pretraining replay is explicitly charged in the ledger, while
target candidate learning and calibration use unique rows. The next gap is
learned temporal delay/absence handling and positive transfer against a fresh
learner; a calibrated reliability gate is not general continual learning.

## Replay-free sufficient-statistics reliability (2026-08-09)

The replay-heavy neural evidence pretraining path now has a narrower
replay-free alternative: `ExternalTransitionEvidenceStatistics`. It maps
generic factual prediction error into bounded logarithmic bins and learns
positive/negative evidence counts from scalar verifier outcomes. Its payload
contains only fixed bin edges, counts, and an observation count; it has no
optimizer, raw examples, or replay buffer. The router consumes it through the
same replaceable evidence-evaluator interface and still applies the explicit
provisional warm-up before allowing reliability to veto continuation.

Across seeds `2101` and `2102`, the statistics evaluator consumed `512` unique
calibration outcomes per seed with zero replay and zero optimizer updates.
Two sequential nonlinear candidates each consumed `64` rows and retained no
raw rows. Clean probability was `0.992` in both seeds, noisy probability was
`0.252` and `0.343`, and corrupted bundles with raw MSE `0.00565` and `0.00439`
were rejected despite a `0.02` router tolerance. Held-out errors remained
below `0.0084`; persistence, candidate isolation, frozen-controller, and
zero-replay gates passed. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_one_pass_evidence_admission_promoted_2026-08-09/`.

This removes evaluator pretraining replay from this reliability boundary, but
it is intentionally not overclaimed: the bins are a bounded calibration
primitive, the audit is sequential rather than interleaved nonlinear routing,
and learned delay/absence behavior and positive transfer remain open.

## Replay-free learned delay and absence policy (2026-08-09)

`EventWaitStatistics` is the first replay-free external state for the
timestamp-buffer decision itself. It consumes only the generic transport
features already exposed to `EventWaitPolicy`—age, present fraction,
completeness, arrival count, and arrival delta—and scalar utility indicating
whether waiting was useful. A fixed quantized main-effect/pairwise basis is
updated through signed ridge sufficient statistics. Exact context counts keep
unseen partial windows at a neutral wait prior instead of extrapolating a
release decision from unrelated evidence. The controller remains frozen and
the buffer uses the same replaceable wait-policy interface.

Across seeds `2301` and `2302`, each run consumed `192` training outcomes and
`8` post-training retention outcomes with zero optimizer updates, zero replay,
and zero raw feature rows. The learned policy waited through a delayed partner
at age one, released a permanently absent partner at age two, released
complete windows immediately, persisted exactly, and retained the earlier
decisions after a new late-absence observation. Block-shuffled outcomes
reversed the learned decisions and failed the control. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_one_pass_wait_statistics_promoted_2026-08-09/`.

This promotes only bounded age/coverage delay and absence handling. It does
not establish natural temporal inference, arbitrary missing-stream reasoning,
positive transfer against a fresh learner, or general continual learning.

## Verifier-gated external program capacity growth (2026-08-09)

The external program router now supports transactional capacity growth. A
growth request first verifies retention on the source state, creates a
copy-on-write candidate with zero-initialized policy and eligibility columns,
and verifies the candidate again. Only after both probes pass does the router
commit the larger address space. New capacity remains inactive until an
explicit append operation activates it, so adding memory cannot silently
change the controller's available action set.

Across seeds `2303` and `2304`, two opaque routes were learned from one-pass
scalar outcomes, capacity grew from two to three, the third route was
activated and learned, and both original routes remained correct. Rejected
growth was a no-write transaction; persistence was exact; reward-shuffled
outcomes failed; and the controller and plasticity rule remained frozen. The
experiment used zero optimizer updates, zero replay, and retained zero raw
feature rows. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_program_capacity_growth_promoted_2026-08-09/`.

## Stable-prefix external program-cell compounding (2026-08-10)

The external program boundary now has a replicated positive-transfer result.
`experiments/external_program_compounding/` learns two opaque executable
program routes into one external source cell, then creates a target cell with a
third program. A copy-on-write challenger gives transferred and fresh target
route states the same `64` current-target rows; the selected cell continues
without replaying source rows. The old source cell remains a separate,
unchanged external memory object.

Using the required stable-prefix metric, all three seeds reached target
mastery in `628` accounted target router updates when warm, versus `1,000`
for matched fresh cells (`1.59x` target acquisition ratio). Old-cell retention
was `97.67%`, `95.67%`, and `98.67%`. The controller, register interpreter,
and program artifacts remained frozen; shuffled outcomes failed; the source
state digest was unchanged by challenger selection; and controller optimizer
updates were zero. The full source-plus-target stable totals were `2,628`
versus `3,000` updates per seed.

This promotes a bounded external program-cell compounding mechanism, not
general continual learning. The source and target novelty support is still
deliberately separable, and an upstream learned context/address path is not
yet demonstrated. The next bottleneck is learned routing among multiple
retained cells under overlapping or drifting evidence. Evidence is archived
in
`session_records/sequence_working_memory_2026-08-02/external_program_compounding_promoted_2026-08-10/`.

## Outcome-routed overlapping external program cells (2026-08-10)

The program-cell boundary now has an append-only bank with stable logical cell
IDs, independently persisted route states, and verifier-gated copy-on-write
selection. `ExternalOutcomeProgramCellBank` probes isolated copies of every
retained cell and reuses one only when its executable predictions explain the
current evidence below the match threshold. A failed match leaves all
committed cells untouched; callers explicitly append a fresh cell.

The pressure test gives two cells the same event features but different hidden
program relations, then alternates the streams. Across three seeds the bank
selected `[0, 1, 0, 1, 0, 1]`, with maximum wrong-cell accuracy below `0.67%`.
Source-cell retention was `82.67–90.33%`, target mastery was `80.67–90.67%`,
shuffled outcomes were rejected, and payload restore reproduced the exact bank
digest. The controller, interpreter, and executable artifacts stayed frozen;
old-cell replay was zero.

This promotes outcome-based routing among overlapping external cells, not
learned context formation from raw modalities or general continual learning.
The next bottleneck is an independently learned address representation that
can handle gradual relation drift and partial evidence without requiring a
complete verifier bundle. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_program_cell_routing_promoted_2026-08-10/`.

This is a promoted bounded external address-space-growth primitive. It does
not establish unrestricted memory growth, arbitrary program induction,
positive transfer against a fresh learner, or general continual learning.

## Imported policy-free learning rule (2026-08-10)

The exported game-learning session provides a useful architectural correction
for interpreting the results above. Across several failed consolidation
mechanisms, forgetting did not disappear when a component was frozen; it moved
into the remaining plastic component. The successful route changed the thing
being stored: an external transition model stores factual predictions of how
an opaque state changes after an opaque intention, while a planner derives
behavior from the current opaque goal at inference time. A fact can be
incomplete on a new regime and be extended by new observations; a stored
preference can be wrong and must be unlearned.

This makes the following priority rule normative:

* The canonical continual-learning path is `event -> factual external model
  -> goal-conditioned search -> intention`. The controller remains frozen
  while external model state learns.
* `ExternalOutcomeProgramRouter` and its program-cell bank are bounded
  preferential-action routing primitives. They are useful controls for
  address growth and delayed credit, but are not the general knowledge
  substrate and do not establish arbitrary new computation.
* New experiments must report deployed capability separately from zero-shot
  capability and acquisition cost. Internal model loss alone is not mastery;
  the stopping prefix must satisfy a held-out planner/verifier gate.
* Every promoted result must begin with measured no-agent and shuffled-outcome
  floors, verify that the requested configuration actually ran, and charge
  actual optimizer work rather than a fixed budget. The exported session found
  multiple plausible results whose harness or metric made the target
  hypothesis impossible to observe.

The session also reinforces a representation constraint: the number-line
control succeeded where a grid formulation stalled because the instruction and
state codes were easier to distinguish. This does not justify hand-assigned
semantics in the controller, but it does require frontends to preserve
separability and experiments to test instruction encoding as a first-class
factor. The next high-ROI work is therefore universal goal-space
generalization, followed by model-first partial/drifting evidence and longer
alternating model-bank growth—not another action-policy consolidation variant.

## Universal opaque-goal search (2026-08-10)

The exported games session exposed a failure mode that finite-goal executor
tests can hide: if only a few goals are presented, an unconditional habit can
score well while ignoring the instruction. The new
`experiments/external_universal_goal_reacher/` audit therefore trains only a
factual external transition model from one-pass opaque state/intention/
next-state observations. It gives the planner no goal labels, then evaluates
`24` held-out goals from `5` starting states each. A finite-goal habit control
sees only nine training targets.

Across seeds `84001`, `84002`, `84003`, and `84004`, the factual model plus
goal-conditioned search reached all `120/120` held-out trials. Goal-shuffled
evaluation and the finite-goal habit both reached `0/120`; random floors were
`0.033`, `0.017`, `0.000`, and `0.017`. The controller and factual model were
unchanged during search, persistence was exact, optimizer updates and replay
were zero, and each search expanded `44,640` nodes.

This audit also fixed a real planner bottleneck. Terminal-only beam search can
prune every useful prefix on a long horizon when intermediate candidates tie.
`ExternalModelBasedPlanner.plan(..., goal_progress_weight=...)` now exposes an
explicit opt-in intermediate heuristic using the same opaque terminal goal
measure. It remains disabled by default because latent-space progress is not
universally meaningful.

This promotes bounded held-out goal-space generalization of behavior derived
from replay-free factual knowledge and the opt-in search heuristic. It does
not establish cross-modal goal abstraction, arbitrary nonlinear goal
representation, unrestricted planning, or general continual learning. Reports
and checksums are archived in
`session_records/sequence_working_memory_2026-08-02/external_universal_goal_reacher_promoted_2026-08-10/`.

## Graded learned goal verification (2026-08-10)

The next rung adds a learned external `ExternalGoalEvaluator`. It receives
only opaque state/goal tensors and deterministic graded scalar verifier
outcomes; the controller remains frozen. The evaluator is trained on nine
coarse noisy goal values and then used by the planner on `24` held-out goal
values with additional noise.

Across seeds `84101`, `84102`, `84103`, and `84104`, deployed mastery was
`0.992`, `0.992`, `0.983`, and `1.000`. Held-out verifier positives were all
above `0.998`, negatives were all below `0.091`, goal-shuffled mastery was
`0.0` on every seed, reward-shuffled evaluator mastery stayed below `0.042`,
and corrupted-goal mastery was `0.0`. The evaluator and factual model were
unchanged during search, persistence was exact, and transition evidence was
consumed once.

This rung also exposed and corrected a planning distinction. A hard binary
goal verifier is useful for terminal acceptance but is too sparse to guide a
long-horizon beam search. Graded verifier outcomes provide an intermediate
goal-progress signal without introducing a task-specific policy.

The evaluator used `648` rows repeatedly for `1,000` offline optimizer
updates, so this is explicitly not replay-free evaluator learning. It promotes
held-out noisy goal verification and external goal-conditioned planning, not
cross-modal goal abstraction or general continual learning. The next
bottleneck is a one-pass or sufficient-statistics goal memory that can migrate
across representation changes without replay. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_learned_goal_evaluator_promoted_2026-08-10/`.

## One-pass sufficient-statistics goal verification (2026-08-10)

The next rung replaces the repeated MLP batch with
`ExternalGoalEvaluatorStatistics`, an independently versioned external memory
that consumes the same graded opaque verifier outcomes once through normal
equation sufficient statistics. Its bounded pairwise distance basis includes
an unbounded distance term for directional search and a clipped term for
robustness to small representation noise; it does not add task labels or a
protocol-specific policy branch.

Across seeds `84201`, `84202`, `84203`, and `84204`, held-out deployed mastery
was `1.000` on every seed. Held-out positive probabilities were at least
`0.958`, negatives were at most `0.107`, goal-shuffled and corrupted-goal
mastery were `0.0`, and reward-shuffled evaluator mastery was `0.0`, `0.225`,
`0.0`, and `0.0`. The controller and factual transition model stayed frozen,
the evaluator was unchanged during search, and exact persistence was verified.

The evaluator consumed `648` unique graded verifier outcomes in one statistics
update, stored no raw rows, replayed zero goal examples, and made zero
controller optimizer updates. This is the first promoted replay-free goal
memory boundary in this ladder, but it is deliberately narrow: a bounded
sufficient-statistics relation is not arbitrary nonlinear goal abstraction,
representation migration, unrestricted memory growth, or general continual
learning. The next bottleneck is migration across changed event/goal bases
without replaying old verifier outcomes. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_one_pass_goal_evaluator_promoted_2026-08-10/`.

## Replay-free goal-memory representation migration (2026-08-10)

The next pressure test changes the frontend representation rather than
retraining the goal memory. A two-dimensional affine replacement frontend is
aligned back into the frozen one-dimensional goal space by
`ExternalGoalRepresentationAlignmentStatistics`, an independent one-pass
normal-equation component trained only on paired replacement/source tensors.
The old verifier outcomes remain in `ExternalGoalEvaluatorStatistics` and are
not replayed.

Across seeds `84301`, `84302`, `84303`, and `84304`, migrated planning reached
`1.000` mastery on every seed. Held-out verifier positives were all at least
`0.999` and negatives were at most `0.032`; shuffled-alignment mastery was at
most `0.058`, missing-alignment mastery was `0.017` on every seed, and
corrupted-goal mastery was `0.0` on every seed. Reward-shuffled evaluator
mastery was `0.0`, `0.0`, `0.0`, and `0.15`.

The alignment consumed `96` paired tensors once, while the old goal memory
consumed its `648` verifier outcomes once. The controller, factual model,
verifier memory, and alignment state were unchanged during search; both
external memories persisted exactly. This promotes a bounded learned
alignment migration path, not arbitrary nonlinear migration, unsupervised
cross-modal grounding, unrestricted memory growth, or general continual
learning. The next bottleneck is evidence-gated handling of nonlinear or
partially observed frontend drift. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_goal_representation_migration_promoted_2026-08-10/`.

## Evidence-gated goal representation drift (2026-08-10)

The alignment boundary now has a non-mutating held-out verification receipt.
Across seeds `84401`, `84402`, `84403`, and `84404`, a partially observed
affine replacement frontend preserved `1.000` planning mastery on every seed;
its maximum held-out alignment MSE stayed below `1e-5`. A genuinely nonlinear
replacement produced held-out MSE between `0.299` and `0.304` and was rejected
on every seed before it could serve the planner or live memory.

The old goal verifier memory stayed unchanged, verifier replay was zero, and
the controller and factual model remained frozen. This closes an important
safety hole: a new representation cannot be promoted merely because it fits
the acquisition pairs. It does not solve nonlinear alignment; the next
bottleneck is a frozen nonlinear external basis or a quarantine-and-grow path
that can acquire one under the same held-out and retention gates. Evidence is
archived in
`session_records/sequence_working_memory_2026-08-02/external_goal_representation_drift_gate_promoted_2026-08-10/`.

## Nonlinear goal representation alignment (2026-08-10)

The rejected-linear-drift result now has a bounded nonlinear successor:
`ExternalGoalRepresentationRandomFeatureAlignmentStatistics` freezes a random
feature basis and learns only its one-pass sufficient statistics. Across seeds
`84501`, `84502`, `84503`, and `84504`, the replacement frontend reached
`1.000`, `0.992`, `0.975`, and `0.992` held-out planning mastery. Nonlinear
alignment MSE stayed below the `0.005` promotion tolerance on every seed,
while the linear candidate remained near `0.30` MSE and was rejected.

The old verifier memory and factual model stayed frozen, verifier replay was
zero, the nonlinear adapter was unchanged during search, and persistence was
exact. Shuffled nonlinear alignment stayed between `0.025` and `0.067`. This
promotes a finite nonlinear alignment basis, not arbitrary nonlinear
computation or unrestricted frontend growth. The next bottleneck is basis
growth or quarantine when held-out evidence exceeds current nonlinear
capacity. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_goal_representation_nonlinear_alignment_promoted_2026-08-10/`.

## Copy-on-write nonlinear goal-alignment growth (2026-08-10)

The nonlinear alignment memory can now grow without replaying its earlier
rows. An initial 16-feature adapter consumed `24` sparse pairs and failed
held-out verification (`0.392–0.625` mastery). A retention-verified
copy-on-write expansion to `80` features consumed only `24` new pairs and
reached `0.950–1.000` mastery across four seeds. Post-growth held-out MSE was
below `0.00284`, and the growth seam's maximum retention error was below
`2e-11` on every seed.

The old alignment and verifier memories remained protected, persistence was
exact, and replay was zero. This promotes bounded external nonlinear capacity
growth, not unrestricted capacity or general continual learning. The next
bottleneck is concurrent frontend pressure: growth refusal, quarantine, and
eviction must preserve multiple nonlinear alignments without silently
forgetting one. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_goal_representation_nonlinear_growth_promoted_2026-08-10/`.

## Seed-widened policy-free model compounding (2026-08-10)

The nested factual-model compounding rung was widened from two to four seeds.
Seeds `70313` and `70314` again reached deployed planner mastery at every
target, beat matched fresh acquisition at every target (`25/28/21` versus
`42/40/31`, and `34/30/23` versus `46/48/43`), retained every prior model at
mastery with byte-stable state, and used zero old-regime replay during target
adaptation. Together with seeds `70311` and `70312`, this is a four-seed
replicated downward acquisition-cost signal.

The claim remains deliberately narrow: one small nested dynamics family,
supplied context vectors, finite planner horizon, and finite external model
capacity. It does not establish general continual learning. The next test
must preserve the same accounting while stressing partial evidence, gradual
drift, or a broader disjoint family. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_transition_model_compounding_seed_widening_promoted_2026-08-10/`.

## Replay-free partial evidence and gradual factual drift (2026-08-10)

The next model-first pressure test now passes on three seeds. Each drift
regime exposed only `8` of `12` available transition rows to an affine
sufficient-statistics model. The router consumed every presented row once,
staged each drift version outside the committed bank, and promoted it only
after held-out prediction and retention verification. Slopes `1.0`, `1.5`, and
`2.0` became stable external slots `[0, 1, 2]`; each reached `1.0` planner
mastery, and returning to the original regime selected slot `0` with `1.0`
mastery and an unchanged digest.

A corrupted stream was staged but rejected by the retention probe without
changing the committed bank. Raw provisional rows, old-regime replay, and
controller updates were all zero; persistence was exact. This is the first
promoted evidence that the factual external boundary can version gradual drift
from partial, replay-free evidence while retaining earlier behavior.

The claim is still bounded: affine dynamics, a fixed opaque context encoder,
finite capacity, and a small planner. It does not establish learned
multimodal context formation, unrestricted growth, nonlinear drift, or general
continual learning. The next bottleneck is learned evidence/address formation
for richer nonlinear drift under the same no-replay and retention gates.
Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_partial_drift_streaming_model_promoted_2026-08-10/`.

## Learned-context nonlinear drift (2026-08-10)

The next three-seed pressure test trains `ExternalTransitionContextEncoder`
only on two source transition bundles, freezes it, and then routes two
nonlinear target drift regimes through `ExternalOnlineTransitionContextRouter`.
The encoder now has a persisted `mean_pool` aggregation mode: it pools
independent learned token features, so the opaque address is invariant to
transport arrival order rather than accidentally depending on the last row.

Each target supplied only `32` of `64` available transition rows. The
random-feature sufficient-statistics slot consumed the four eight-row windows
once, and promotion required held-out factual prediction plus source-slot
retention. Seeds `82001`, `82002`, and `82003` all passed, with target held-out
MSEs of `1.31e-4/3.57e-4`, `6.70e-4/3.27e-4`, and
`1.51e-3/9.83e-5` for target C/D. Source re-routing returned to the original
slot; corrupted evidence staged but failed held-out verification without a
bank write; no raw provisional rows or old-regime examples were replayed; the
controller remained frozen; and router persistence was exact.

This promotes bounded replay-free nonlinear drift retention with a
source-trained permutation-invariant learned address. It does not establish
unrestricted memory growth, arbitrary new computation, or general continual
learning: the context encoder is pre-trained before target exposure, the
model basis is fixed and finite, and the stream has only two target regimes.
The next bottleneck is online address adaptation under distribution shift
without changing the meaning of old keys, followed by verified model-bank
growth and compression under longer alternating streams. Evidence is archived
in
`session_records/sequence_working_memory_2026-08-02/external_nonlinear_drift_learned_context_promoted_2026-08-10/`.

## Copy-on-write learned address adaptation under long nonlinear alternation (2026-08-10)

The next boundary adds `ExternalTransitionContextAddressAdapter`. It keeps
the source-trained context encoder and every committed bank key immutable. A
factual mismatch creates an isolated address-encoder copy; the copy learns
stability and generic separation from historical keys using only the current
evidence window. Its version is committed only alongside a held-out,
retention-verified factual model candidate.

The three-seed stream acquired four nonlinear target regimes from `32/64`
presented rows each, interleaved source and target revisits across twelve
streams, and ended with six retained slots. Every revisit matched an existing
slot; no duplicate target slot was minted. Address versions advanced to `20`
per seed while historical keys stayed unchanged. Corrupted evidence staged but
failed held-out verification without a bank write; controller updates, raw
candidate retention, and old-regime replay were zero; persistence was exact.
All gates passed for seeds `82101`, `82102`, and `82103`.

This promotes bounded long-horizon nonlinear factual-memory routing with
copy-on-write learned address versions. It does not establish open-world
continual learning: the factual basis and capacity are finite, the encoder is
source-pretrained, and address adaptation is triggered by a router-detected
mismatch. The next frontier is unbounded growth with verified consolidation or
compression, plus a learned evidence policy that can decide when a mismatch is
real under noisy multimodal streams. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_nonlinear_address_shift_stream_promoted_2026-08-10/`.

## Verifier-gated nonlinear address-space growth (2026-08-10)

The same long alternating stream now starts with a four-slot external bank.
Before admitting the fifth regime, `ExternalOnlineTransitionContextRouter`
calls `grow_verified(6, retention_probe)`. The transaction changes capacity
metadata only after pre/post held-out retention checks; it does not rewrite
contexts, models, or address versions.

Seeds `82101`, `82102`, and `82103` all promoted the `4 -> 6` growth, acquired
the remaining nonlinear regimes, revisited every target successfully, and
passed the existing copy-on-write address, historical-key immutability,
corruption, frozen-controller, zero-replay, and exact-persistence gates. This
promotes verified nonlinear memory growth under learned address adaptation, not
unrestricted growth or autonomous consolidation/compression. Evidence is
archived in
`session_records/sequence_working_memory_2026-08-02/external_nonlinear_address_shift_capacity_growth_promoted_2026-08-10/`.

## Nonlinear address compaction and statistics-aware codec boundary (2026-08-10)

The next three-seed lifecycle audit composes the long nonlinear address stream
with two external-memory operations. Capacity grows `4 -> 6 -> 7` only after
held-out retention probes. A copy-on-write equivalent of the `target_c` factual
slot is then added, and `consolidate_verified()` shares its parameters with the
original while preserving both opaque logical addresses. Physical models fall
from `7` to `6`; historical model digests and address keys remain unchanged.

The audit tests legacy float16/int8, row-int8, and the new
`float16_stats` codec against a baseline-plus-`1e-3` held-out factual
retention delta. The legacy codecs are rejected: quantizing the replay-free
random-feature normal-equation statistics produces deltas as high as `1.1233`
for float16 and `6.5146` for int8. `float16_stats` preserves the immutable
Fourier basis and ill-conditioned normal matrix, stores the solved predictor in
float16, and reconstructs the target matrix on restore. It is selected on all
three seeds, reducing bank storage from `487,564` to `483,952` bytes while
keeping every held-out delta below `5e-5`. Alias and logical-address
round-trips remain exact.

Controller updates, compaction optimizer updates, replay, and old-regime
replay are zero; corruption is staged and rejected without a bank write; and
router persistence is exact for all three seeds. This promotes bounded
retention-verified factual lifecycle management and statistics-aware storage
compression, not semantic merging, unrestricted memory growth, or general
continual learning. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_nonlinear_address_compaction_promoted_2026-08-10/`.

## Open-world-style address formation without encoder pretraining (2026-08-10)

The next pressure test removes the source-pretraining assumption from learned
address formation. `ExternalTransitionContextEncoder` starts untrained and
receives zero optimizer updates. Eight nonlinear regimes arrive sequentially;
capacity grows `1 -> 8` through seven retention-verified transactions, and
each new address is formed by isolated copy-on-write adaptation from its
current evidence windows.

Seeds `82401`, `82402`, and `82403` all passed. Each acquired eight distinct
slots, reached held-out MSE below `0.004`, returned to every regime in reverse
and interleaved order with matched existing slots, retained all earlier factual
models, rejected corrupted evidence without a bank write, kept the controller
frozen, retained no raw candidate rows, and restored exactly. Address versions
advanced to `40` per seed; context-encoder pretraining and old-regime replay
were both zero.

This is stronger than source-pretrained learned-context routing: identity is
formed online by an external copy-on-write state while historical keys remain
immutable. The claim remains bounded: the stream and capacity are finite, the
factual basis is fixed random features, and this does not establish
unrestricted general continual learning. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_open_world_address_stream_promoted_2026-08-10/`.

## Partial and ambiguous open-world evidence (2026-08-10)

The next integrated pressure test combines online identity formation with
partial, concurrently arriving, and explicitly ambiguous evidence in
`experiments/external_partial_ambiguous_open_world/`. Four nonlinear regimes
arrive with only `32/64` available training rows. The context encoder starts
untrained and remains at zero optimizer updates; identity is formed by the
external copy-on-write address adapter, and factual slots use one-pass
random-feature sufficient statistics.

Two novel regimes are staged concurrently. An eight-row bundle whose factual
predictions are indistinguishable is held in bounded quarantine outside every
candidate. A later opaque factual-routing decision anchors the current stream
to one candidate; the router then consumes the quarantined bundle exactly once
in the same adaptation transaction. Promotion is refused while quarantine is
unresolved, so ambiguous evidence cannot silently become committed memory.

Seeds `82501`, `82502`, and `82503` all passed the partial-evidence,
quarantine-and-resolution, four-slot growth, held-out factual promotion,
alternating-revisit, prior-retention, corruption, copy-on-write, frozen
controller, zero-replay, and exact-persistence gates. Maximum held-out MSE was
`0.00109`, `0.00107`, and `0.00287`; each seed quarantined and then consumed
exactly eight rows.

This promotes a bounded replay-free nonlinear factual-memory identity boundary
under partial and explicitly ambiguous evidence. It does not establish
unrestricted continual learning: the stream and capacity are finite, the
factual basis is a fixed random-feature family, the ambiguity control is
verifier-constructed, and the controller still has no learned multimodal
context formation or arbitrary new computation. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_partial_ambiguous_open_world_promoted_2026-08-10/`.

## Rejected naive learned-MLP factual substitution (2026-08-10)

The new `streaming_gradient` protocol was applied to a trainable
`ExternalTransitionModel` MLP in
`experiments/external_learned_nonlinear_open_world/`. Four regimes exposed
only `48/64` training rows. The controller and context encoder stayed frozen;
raw provisional rows and old-regime replay were zero. Current four-row
windows received four local optimizer updates, and those current-window
reuses were counted separately.

The model sometimes passed its current held-out factual gate, but three seeds
could not reliably route later revisits to the old logical slots. A strict
factual routing tolerance caused capacity pressure instead of a match; a loose
tolerance prematurely matched novel regimes. Seed `82602` also failed the
stricter `0.08` held-out quality gate. The result is rejected, with the fixed
random-feature sufficient-statistics family retained as the current baseline.

This rejects the naive substitution, not learned nonlinear factual memory in
general. A successful replacement needs a representation-stable or
meta-learned initialization and an independently verified route query; lower
training loss alone is insufficient. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_learned_nonlinear_open_world_rejected_2026-08-10/`.

## Rejected frozen trajectory-statistics route query (2026-08-10)

The exported game-learning session's successful `trajectory_stats` idea was
ported into a versioned `ExternalTransitionRouteQuery`. It proposes opaque
slots from slot-local copy-on-write address adapters and stores only a richer
route key: projected context, final recurrent state, mean recurrent state,
and max recurrent state. A proposal-quality floor was added, but factual
prediction verification remained independent and strict.

Across seeds `82601`, `82602`, and `82603`, the learned MLP acquired all four
partial-evidence regimes, retained the controller byte-for-byte, stored no
raw candidate rows, replayed no old evidence, and restored exact route state.
However, revisit identity matched only `0/6`, `1/6`, and `0/6` queries; one
seed also failed held-out model quality. A loose factual tolerance admitted a
novel regime during smoke and is rejected as a workaround.

The route query sometimes nominated the correct slot, but a frozen similarity
metric did not provide reliable identity calibration. This rejects threshold
tuning and frozen cosine routing for learned nonlinear open-world memory. The
next candidate must learn a reusable route score from verifier-grounded
counterfactuals or a meta-learned representation, while retaining a factual
challenger and explicit novel-regime rejection. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_learned_nonlinear_route_query_rejected_2026-08-10/`.

## Rejected shared current-window learned route scorer (2026-08-10)

The next candidate reused the repository's permutation-equivariant
`OpaqueCandidateGrowthRouter` as a trainable external route scorer. It scored
opaque trajectory-statistics queries against opaque slot keys and learned from
paired factual counterfactual utilities. The controller and base context
encoder remained frozen; route updates used only the current evidence window,
with no old-regime replay or raw transition rows retained.

Across seeds `82601`, `82602`, and `82603`, the scorer repeatedly collapsed
toward the newest slot. Route proposals failed the factual-winner diagnostic
and revisit identity remained `0/6` for every seed. Exact route-state
persistence and corruption rejection passed. A factual fallback was added and
regression-tested so a bad proposal cannot override a verifier-established
match, but that safety fallback is not a learned capability gain.

This rejects a shared scorer trained only on the newest window. The failure
shows that the route learner itself needs isolated per-slot state or a
compressed verifier-maintained route-constraint memory; more updates on the
same window and threshold tuning would merely reinforce forgetting. Evidence
is archived in
`session_records/sequence_working_memory_2026-08-02/external_learned_nonlinear_learned_route_query_rejected_2026-08-10/`.

## Slot-local prototype route memory (2026-08-10)

The next boundary adds `ExternalTransitionRouteMemory`, a bounded external
store of normalized opaque trajectory prototypes owned by stable logical
slots. Verified matches may merge or append state only to the winning slot;
the controller, shared route scorer, and historical slots remain untouched.
The store contains no raw transition rows, persists independently, and is
still only a proposal mechanism behind factual verification.

Across seeds `82601`, `82602`, and `82603`, route-memory state restored
exactly, corruption rejection and safe factual fallback passed, and prototype
updates remained external with zero old-regime replay. However, route
proposals failed the factual-winner diagnostic on every seed and revisit
identity remained `0/6`. The untrained nonlinear context representation made
distinct regimes look too similar, so the memory preserved insufficient
information rather than inventing it.

This retains slot-local prototype memory as the correct continual-state
boundary but rejects it as a standalone route-capability gain. The next
pressure test must make the representation stable or meta-learned before
freezing it; threshold tuning and additional prototypes are not substitutes
for missing identity information. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_transition_prototype_route_memory_rejected_2026-08-10/`.

## Rejected verified transfer-vs-fresh nonlinear prior (2026-08-10)

The next pressure test used isolated copy-on-write challengers to decide
whether a novel nonlinear factual-model candidate should start from a prior
slot or a fresh model. Both challengers were trained only on the current
bundle; the lower factual probe loss selected the candidate, and the
controller, committed slots, raw provisional evidence, and old-regime replay
remained unchanged. The selection receipt persisted exactly and the probe was
covered by a source-state isolation test.

Across seeds `82601`, `82602`, and `82603`, verified transfer was selected for
every novel regime after the first. It did not improve the promoted gates:
held-out quality passed `1/3` seeds for the verified arm versus `2/3` for the
automatic-prior control, and revisit identity remained incomplete (`0/6`,
`2/6`, and `2/6`). One seed improved, one partially improved, and one
regressed. The factual fallback remained correct, but that safety behavior is
not a capability gain.

This rejects transfer-prior selection as the missing mechanism. A local
challenger can choose a better initialization, but it cannot create a stable
identity representation for nonlinear regimes. The exported session's larger
lesson therefore stands: factual transition models plus goal-conditioned
search are the durable substrate; stored policies, adapters, and learned
initializations are preference-shaped state that can still become stale. The
next experiment must meta-learn or otherwise stabilize the factual model's
representation and route identity, while preserving verifier-gated
copy-on-write and no-replay accounting. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_learned_nonlinear_verified_transfer_prior_rejected_2026-08-10/`.

## Promoted replay-free nonlinear factual-memory retention (2026-08-10)

The nonlinear factual-memory pressure test now uses sixteen local optimizer
updates per four-row current evidence window rather than four. Across seeds
`82601`, `82602`, and `82603`, all four `48/64` partial-evidence regimes
passed held-out acquisition, all six revisits returned the correct stable
logical slot, all prior slots remained byte-stable, corruption was rejected
without a bank write, and exact persistence passed. The controller and
context encoder received zero optimizer updates; old-regime replay and raw
provisional-row retention were zero.

This promotes bounded factual-model retention, not route memory. The router
identified slots through factual transition prediction; route-query and
prototype-memory updates were zero. Separate prototype audits still show
that opaque route proposals can be wrong, so proposals remain optional
accelerators behind factual verification. The improvement is therefore
attributed to better current-window model fitting, not to a routing shortcut.

The result remains bounded to four synthetic nonlinear regimes, finite model
capacity, supplied opaque transition bundles, and sixteen-step local fitting.
It does not establish unrestricted growth, arbitrary new computation, or
general continual learning. The next test should use a genuinely different
dynamics family or partial/gradual drift while preserving the same retention,
fresh-control, replay, and optimizer accounting. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_learned_nonlinear_factual_memory_16step_promoted_2026-08-10/`.

## Route-representation diagnostic rejection (2026-08-10)

Source-only context pretraining, direct trained context-key route features,
and an eight-regime meta-pretrained recurrent trajectory feature were tested
against the nonlinear route-memory fixture. On seed `82601`, route proposals
matched factual winners only `2/6`, `1/6`, and `2/6`, respectively, and every
arm had `0/6` factual revisits under the four-step model-fitting budget.
Held-out acquisition still passed, so the failure is specifically route
identity rather than basic model fitting.

This rejects the route-representation variants as capability gains, while
retaining their explicit feature-space boundary and source-only training
hooks. The result reinforces the distinction between factual retention and
route acceleration: route proposals must remain non-authoritative until a
verifier-grounded factual signature can reliably identify a slot. Evidence is
archived in
`session_records/sequence_working_memory_2026-08-02/external_nonlinear_route_representation_diagnostics_rejected_2026-08-10/`.

## Factored factual computation with external residual memory (2026-08-10)

The next implementation separates reusable factual computation from
context-local adaptation. ExternalFactoredTransitionModel trains a shared
transition base once, freezes it, and stores only opaque context-addressed
residual facts in an append-only external memory. The planner derives
intentions from the sum of base prediction and an exact residual hit; no task
policy is stored in the base or residual memory.

Across seeds `82701`, `82702`, and `82703`, four genuinely disjoint transition
regimes were presented sequentially. Source regimes received complete
evidence; each target received only a verifier-private target-covering subset
(`5` or `7` of `14` rows). All planner goals reached `1.0` mastery, every
earlier regime retained mastery after later writes, the base and context
encoder remained byte-stable, residual adaptation used zero optimizer updates,
and exact persistence passed.

This promotes a bounded factored factual-memory boundary under partial
evidence. It does not establish automatic context formation, arbitrary
missingness, unbounded residual growth, compression, or general continual
learning. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_factored_transition_residual_promoted_2026-08-10/`.

The router also exposes a bound-stream copy-on-write update path. Once an
opaque slot is already owned by a stream binding, a new fact can be staged on
an isolated model copy without asking the identity router to rediscover the
slot. The update is committed only after a separate held-out factual
observation and a caller-owned retention probe pass; the API deliberately
does not permit the admitted row to serve as its own held-out gate. This is a
reusable safety boundary, not yet a promoted claim about automatic online
learning.

## Open-set binding and retention-safe lifecycle (2026-08-10)

`ExternalOnlineStreamBindingMemory` now has an explicit open-set lifecycle.
When live capacity is full, evidence that does not match a live anonymous
track enters bounded provisional memory. The result carries only a provisional
ID; it cannot expose a live key or reach the factual router. Subsequent
arrivals update only that provisional's bounded observations, prototype,
delay estimate, and verifier sufficient statistics. Provisional state is
therefore useful evidence without becoming an authority by accident.

Admission and retirement are copy-on-write transactions. A caller-owned
retention probe must approve a provisional promotion or live-track retirement;
rejected probes leave the complete binding state unchanged. The serialized
state includes live tracks and provisional tracks with schema/versioned
configuration and recursive tensor checksums, so a restart preserves the
quarantine boundary exactly.

The open-set pressure test in `experiments/external_learned_stream_binding/`
uses four anonymous streams with capacity for three live tracks. Across seeds
`2301` and `2302`, the fourth stream remained provisional through six arrivals
with irregular timestamps, failed admission and retirement probes preserved
live state, and verified retirement followed by promotion admitted it while
the sibling tracks remained intact. All gates passed and persistence was
exact. The encoder and controller were frozen during deployment; provisional
updates used six fresh verifier outcomes and zero replay.

This promotes a bounded learned open-set transport lifecycle, not open-ended
identity discovery, learned eviction policy, arbitrary drift handling,
unrestricted memory growth, or general continual learning. The next pressure
point is to learn the retention/admission evidence policy and to stress
multiple simultaneous provisional identities, contradiction, drift, and
capacity pressure without caller-designed probes.

For unbound input, `ExternalFactoredTransitionRouter.route_bundle` provides
the safer atomic boundary: the full opaque evidence bundle is compared with
all retained factual slots before novel evidence is staged. This avoids
letting a single shared transition decide identity and avoids combining
interleaved novel streams in one global pending window. Row-wise observation
remains appropriate only after a caller has an established stream binding.

## Learned external residual functions (2026-08-10)

The factored model now has an optional learned residual-function backend. It
keeps the shared transition base frozen and places one replaceable external
learner behind each opaque context. The existing exact residual-memory mode
remains the default compatibility path; the learned mode can use the
repository's nonlinear MLP or affine sufficient-statistics family. Candidate
training occurs on a copy and promotion still requires an independent
held-out factual observation plus retention of every prior slot.

The three-seed pressure test in
`experiments/external_factored_learned_residual/` used a nonlinear shared
source base, partial `16/20` online evidence, and an affine context-local
residual family. All seeds promoted both regimes, routed six alternating
revisits correctly, rejected a corrupted bound update without changing the
committed digest, preserved the frozen controller/base/context encoder, and
restored exact state. Target held-out MSE improved over the frozen-base-only
control on all three seeds. A fresh-target learner won on one seed and lost on
two, so the result promotes a generalizing external factual adaptation
boundary, not positive transfer or general continual learning.

Accounting recorded zero old-regime replay during target adaptation and zero
controller optimizer updates. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_factored_learned_residual_promoted_2026-08-10/`.

The next bottleneck is to make the learned residual representation robust on
genuinely nonlinear, partially observed, and drifting dynamics while keeping
the independent gate and fresh-control comparison. A model that only fits a
small affine fixture is not yet the general factual learner required by the
architecture.

## Replay-free nonlinear drift retention (2026-08-10)

The factored residual boundary now also accepts the replay-free random-feature
sufficient-statistics family. Its feature projection is frozen; adaptation
updates only external normal-equation statistics, so the controller, shared
base, and context encoder remain unchanged and each new transition is consumed
once. The random-feature width, seed, ridge, and learning-family selection are
persisted as part of the external model contract.

The pressure test in `experiments/external_factored_nonlinear_drift/` presents
an opaque nonlinear source regime, a partially observed nonlinear target, and
a later drift on the already-bound target slot. Five seeds promoted the source,
target, and drift slots; each passed independent held-out gates, retained the
prior target behavior, routed alternating source/target bundles as
`[0, 1, 0, 1, 0, 1]`, rejected a corrupted drift update without mutation, and
round-tripped exact state. Drift held-out MSE beat the frozen-base-only control
on every seed, with no old-regime replay and no controller updates.

This is a promoted bounded result, not general continual learning. The basis
is fixed and finite, the evidence is smooth synthetic data, and the route
tolerance is an explicitly calibrated post-promotion read setting. The
optimizer-based nonlinear MLP variant was rejected under the same sparse
no-replay pressure because it did not reliably beat the frozen-base control;
that negative result supports retaining additive sufficient-statistics memory
as the current replay-free default for this boundary.

Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_factored_nonlinear_drift_promoted_2026-08-10/`.

## Factored external-memory lifecycle (2026-08-10)

The canonical factored router now owns the external residual bank's lifecycle
boundary. A learned residual model may declare a bounded capacity; the router
keeps that capacity synchronized with its admission limit, exposes
retention-verified growth, delegates independently gated storage-compression
selection, and evicts by stable logical slot ID while repairing its local route
cache. The router and bank therefore cannot silently disagree about whether a
slot exists or which physical index represents it.

The five-seed pressure test in
`experiments/external_factored_memory_lifecycle/` admitted two regimes at
capacity two, grew to four, admitted two more, selected and round-tripped
`float16_stats` compression, evicted middle slot `1`, and admitted a fifth
regime as slot `4`. Every seed preserved and routed all surviving regimes,
including after persistence. No controller, base, or context-encoder updates
or old-regime replay were used.

This removes a concrete implementation bottleneck, but it is not automatic
memory management. Capacity growth, compression, and eviction are still
caller-selected and verifier-gated; the next gap is learned context/version
formation under missing, contradictory, and genuinely open-world evidence.

Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_factored_memory_lifecycle_promoted_2026-08-10/`.

## Partial and contradictory read safety (2026-08-10)

The factored router now has a separate read-only `route_partial_bundle` path.
It requires a configurable fraction of rows to agree with a known factual
slot, applies an explicit contradiction floor, and never stages or writes a
new slot. Empty evidence returns an explicit ambiguous no-op. This separates
missing evidence from admission and prevents a contradictory mixture from
being accepted merely because its mean error favors one existing version.

Across five seeds, five-row partial reads routed known regimes correctly;
mixed-regime evidence was ambiguous; empty evidence was a no-op; and the
router digest remained unchanged. The result is archived in
`session_records/sequence_working_memory_2026-08-02/external_factored_partial_read_safety_promoted_2026-08-10/`.

This closes a read-safety gap, not the full version-formation problem. The
system still needs a learned policy for when unresolved evidence should be
quarantined, revisited, merged into an existing version, or promoted as a new
version under genuinely missing streams and nonstationary context.

## Replay-free partial-stream acquisition (2026-08-10)

The factored row-wise candidate path now has a promoted bounded result under
short current streams. Across five seeds, four nonlinear regimes each staged
after seven of fourteen stream rows, learned from the remaining current rows
through replay-free random-feature sufficient statistics, and passed an
independent four-row held-out gate. Later full revisits and one-row partial
reads routed correctly; contradictory and empty reads remained read-only and
ambiguous. The controller, base, and context encoder stayed frozen.

This shows that the strongest current factual learner can acquire a new
version from a short stream without replay. It does not solve the rejected
short-prefix context-identity problem: the context encoder is still frozen,
the stream schedule is fixed, and open-world version formation under unseen
missingness remains unqualified. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_factored_partial_stream_promoted_2026-08-10/`.

## Unresolved-evidence quarantine (2026-08-10)

The factored router now has an explicit bounded quarantine for evidence that
cannot yet be safely routed. Quarantined bundles are copied as separate
external records, persist through the router payload, can be inspected without
mutation, and are released only by an explicit caller action. They never train
or promote a candidate automatically. The five-seed recovery audit retained,
reloaded, inspected, and released four contradictory one-row bundles per seed
without merging their boundaries.

This prevents the safety path for missing evidence from becoming a data-loss
path. It still needs a learned resolver that can use later evidence to decide
whether a quarantined bundle belongs to an existing version or a new one.
Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_factored_quarantine_recovery_promoted_2026-08-10/`.

The first recovery policy is now implemented as a factual resolver: it
re-tests each quarantined bundle independently against committed slots, removes
only independently matched bundles, and leaves corrupted or unresolved
bundles isolated. Five seeds resolved known bundles to `[0, 0, 1, 1]` and
retained a corrupted bundle without any model write. This is a safe recovery
primitive, not the learned open-world resolver still required by the
architecture. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_factored_quarantine_resolver_promoted_2026-08-10/`.

The resolver now also handles a later-evidence candidate path. If a novel
bundle is quarantined before a subsequent bundle stages a copy-on-write factual
candidate, `resolve_quarantine_to_candidate()` tests the retained bundle
against that isolated candidate and consumes it once only when its predictions
agree. Five seeds passed this continuation, exact candidate persistence, and
committed-model byte stability. The candidate remains unpromoted until a
separate held-out factual and retention gate passes. This closes a concrete
quarantine credit-accumulation hole; it is not the learned open-world identity
resolver or automatic new-version policy still required by the architecture.
Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_factored_quarantine_candidate_resolution_promoted_2026-08-10/`.

The factored router can optionally attach the persistent
`ExternalSparseTransitionEvidenceIndex`. It stores unique opaque factual
input/output overlaps with running means and proposes a slot only when enough
overlaps agree and none contradict. It is a proposal accelerator rather than a
learned semantic identity system; partial-read and promotion gates remain the
authority, and the index must be invalidated with its logical slot.

The first randomized-missingness pressure test for this boundary also passes:
five seeds presented four nonlinear regimes through two disjoint random
seven-row windows. All random partial reads routed correctly, mixed-regime
evidence remained ambiguous, and the sparse index persisted `56` unique facts
per seed. The result is bounded synthetic missingness handling; it does not
establish arbitrary real multimodal missingness or learned semantic identity.
Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_factored_random_missingness_promoted_2026-08-10/`.

The sparse index was then tested during nonlinear drift. Five seeds retained
the source, target, and six drift facts as `46` conflict-preserving opaque
records per seed; source/target alternation, drift held-out quality, corrupted
update rejection, and exact persistence all passed. Same-input contradictory
outcomes are kept as separate factual versions rather than averaged away.
This is bounded replay-free drift retention, not arbitrary nonstationarity or
learned semantic identity. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_factored_nonlinear_drift_sparse_identity_promoted_2026-08-10/`.

## Composition: randomized partial evidence plus drift (2026-08-10)

The two strongest factored external-memory mechanisms were composed in
`experiments/external_factored_random_drift/`. Four opaque nonlinear regimes
were admitted through independent randomized seven-row windows. Each retained
slot then received a disjoint randomized drift update containing only `4/8`
rows, while the other four rows served as an independent drift gate. The
previous regime's held-out rows were used as a retention gate and were not
replayed during the drift transaction.

Across five seeds, every initial and drift version promoted. Random partial
reads routed the correct opaque slot, mixed-regime evidence remained
ambiguous, all retained initial and drift held-outs stayed below the prediction
tolerance, and router state plus the sparse evidence index round-tripped
exactly. The controller, shared base, and context encoder remained byte-stable;
old-regime replay and optimizer updates were zero. Each seed consumed `56`
unique initial observed rows, `16` unique drift-update rows, and performed `12`
external residual-statistics updates.

This promotes only a bounded composition result: replay-free randomized
partial evidence plus gradual factual drift. It does not establish learned
semantic identity, arbitrary open-world version formation, nonstationary
distribution discovery, unrestricted memory growth, or general continual
learning. The unresolved high-ROI problem remains a learned resolver that can
decide when later evidence belongs to an existing version, a new version, or
an unresolved quarantine without caller-provided stream identity.

Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_factored_random_drift_promoted_2026-08-10/`.

## Autonomous verifier-gated capacity growth (2026-08-10)

The online and factored routers now expose an opt-in `auto_grow` lifecycle
mode. When a novel candidate arrives at capacity, the router may stage it
instead of returning capacity immediately. It creates the growth metadata only
on the isolated candidate and commits the larger capacity together with the
candidate, after the independent held-out factual check and caller-owned
retention probe pass. A rejected candidate leaves both live content and
capacity unchanged. Payloads persist the mode, and legacy payloads default to
the previous manual-growth behavior.

The untrained-encoder open-world stream was rerun with this mode. Across three
seeds, eight nonlinear regimes formed eight distinct slots while capacity grew
automatically from `1` to `8`; reverse and interleaved revisits matched existing
slots, all held-out errors passed, corrupted evidence was rejected without a
bank write, and exact persistence held. The context encoder received zero
pretraining updates, the controller stayed frozen, and old-regime replay was
zero. The factored router exposes the same transaction for its residual
memory, so capacity growth is no longer a caller-only precondition for either
canonical factual path.

This promotes bounded autonomous external-memory growth, not unrestricted
memory growth, learned eviction, arbitrary computation, or general continual
learning. The remaining open-world bottleneck is still learned evidence
formation under broader, noisy, multimodal streams rather than a fixed
synthetic transition family.

Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_open_world_auto_growth_promoted_2026-08-10/`.

## Recursive deployed-rollout verification (2026-08-10)

The exported session highlighted a measurement hazard: a transition model can
look good under one-step loss while its predictions compound into a bad
planner. The canonical `ExternalModelBasedPlanner` now exposes a versioned
`ExternalTransitionRollout` probe. It is verifier-owned, accepts only opaque
state/intention tensors and optional external context, recursively feeds each
prediction into the next step, and reports confidence-weighted held-out state
error. The probe is never written to model memory.

The policy-free compounding audit uses this boundary as a promotion gate in
addition to deployed planner mastery and retention. Across three seeds every
adapted target passed recursive rollout error below `0.05`, while the controller
remained frozen, old-regime replay remained zero, and prior model bytes stayed
stable. Zero-shot rollout error is reported separately so new knowledge is not
confused with post-adaptation mastery.

This promotes a reusable verification boundary and a bounded nested-dynamics
result. It does not establish general continual learning, unrestricted memory
growth, or arbitrary multi-step transfer. The next meaningful pressure test is
to apply the same recursive gate to broader, noisy, partially observed dynamics
and to candidate promotion—not only to already selected model slots.

Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_transition_model_compounding_rollout_verified_promoted_2026-08-10/`.

## Recursive rollout as a candidate-promotion invariant (2026-08-10)

The recursive rollout probe is now accepted directly by both canonical factual
promotion boundaries: `ExternalOnlineTransitionContextRouter` and
`ExternalFactoredTransitionRouter`. When supplied, a candidate must pass its
one-step held-out observation, recursive held-out rollout, and retention probe
before its model, address, slot, or capacity metadata can reach live state.
The rollout executes against an isolated candidate bank/model and an opaque
transition context; it never becomes model memory.

The three-seed pressure test promoted a source and target affine regime,
automatically grew capacity from `1` to `2` only after the target passed the
recursive gate, and rejected a candidate with a valid one-step fit but a
corrupted later rollout. Rejection left live content and capacity unchanged.

This is a stronger promotion invariant, not a claim of general continual
learning. The next bottleneck is the same gate under noisy, partial,
contradictory nonlinear evidence, where candidate identity and rollout
verification must remain reliable without replaying old observations.

Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_recursive_candidate_promotion_verified_2026-08-10/`.

## Recursive promotion under partial and ambiguous nonlinear evidence (2026-08-10)

The recursive promotion invariant was applied to the existing harder
open-world stream rather than only to the affine smoke test. Four nonlinear
regimes arrived through partial windows; two candidates were isolated
concurrently, a contradictory bundle was quarantined outside candidate state,
and later opaque factual evidence resolved it once. Every candidate promotion
also passed a three-step verifier-owned rollout probe.

Across three seeds, maximum recursive rollout error was `0.000845`, `0.001634`,
and `0.000526`, under the `0.003` gate. The controller and context encoder
remained frozen, model updates were streaming sufficient-statistics updates,
old-regime replay was zero, prior slots remained byte-stable, and corruption
was rejected without a bank write.

This strengthens the bounded claim from “partial identity plus one-step
promotion” to “partial identity plus recursive promotion verification.” It
still does not establish general continual learning: the evidence is synthetic,
the nonlinear basis is fixed, the stream is finite, and noisy multimodal
candidate formation remains the next bottleneck.

Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_partial_ambiguous_open_world_rollout_verified_2026-08-10/`.

## Robust inlier routing for sparse noisy evidence (2026-08-10)

The online factual router now has an explicit opt-in robust aggregation mode.
`minimum_inlier_fraction` and `outlier_tolerance` are versioned configuration,
persist through checkpoints, and apply to committed-slot routing and provisional
candidate continuation. The default remains the legacy mean-error behavior for
checkpoint compatibility. In robust mode, a candidate is eligible only when
enough rows fall within the configured inlier tolerance; the route score is
computed over those inliers, so a bounded sparse outlier cannot dominate while
a larger contradiction cannot be silently averaged into identity.

The nonlinear partial/ambiguous stream was rerun across three seeds with a
`0.75` inlier fraction and `0.5` outlier tolerance. A deliberately corrupted
row in a partial revisit reused the existing slot on every seed without
capacity growth or candidate staging. The prior quarantine, recursive rollout,
retention, corruption, frozen-controller, and zero-replay gates remained true.

This promotes robust routing infrastructure under a fixed threshold, not
learned noise adaptation. The next bottleneck is learning reliability or delay
from verifier outcomes while preventing that plastic state from changing old
identities.

Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_partial_ambiguous_noisy_robust_rollout_verified_2026-08-10/`.

## Separated learned reliability and delay state (2026-08-10)

The next boundary is now promoted across three seeds. A replay-free
`ExternalTransitionEvidenceStatistics` component consumes scalar verifier
outcomes and is connected to an opt-in committed-slot veto. The veto is
read-only with respect to factual memory: it may reject an evidence route, but
cannot change a historical model, context key, or stable slot ID. A low-error
corrupted revisit was inside the factual match tolerance, so a fresh
gate-disabled control matched it; the learned gate rejected it without a bank
write, and a later clean reversal reused the original slot.

The same audit persisted `EventWaitStatistics` separately. It learned a high
wait probability (`0.999665`) for delayed incomplete evidence and a low wait
probability (`0.000335`) for fast absence. Each seed consumed 128 reliability
and 128 wait outcomes once, with zero replay, zero controller updates, and
exact persistence of both external states and the router configuration.

This promotes a bounded separation of factual identity, learned evidence
reliability, and learned delay policy. It does not establish unrestricted
memory growth, multimodal reliability grounding, arbitrary new computation,
or general continual learning. The next pressure test should make reliability
and delay operate online during multiple concurrent nonlinear streams, with
reversal and missingness interleaved rather than calibrated in a separate
fixture.

Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_reliability_delay_promoted_2026-08-10/`.

## Online interleaved reliability and delay (2026-08-10)

The separated boundary now learns online rather than from a separate
calibration fixture. Two factual streams were interleaved across three seeds.
After four clean verifier outcomes, the committed-slot reliability gate
vetoed a low-error corrupted revisit in stream B without staging a replacement
candidate; later clean observations from both streams routed back to their
original stable slots. High-error novel evidence remained eligible for
candidate formation when capacity was available. A fresh gate-disabled control
matched the same corruption, establishing a causal reliability effect rather
than a tolerance artifact.

Incomplete timestamp evidence was updated in the same run. The wait policy
learned a `0.999665` probability for delayed incomplete evidence and
`0.000335` for fast absence. The controller and factual bank remained
byte-stable, router and external-state persistence was exact, and replay and
post-source factual model updates were zero across all seeds.

This promotes bounded online interleaved reliability/delay state. It still
does not establish learned multimodal grounding, unrestricted memory growth,
arbitrary new computation, or general continual learning. The next pressure
test should couple this state to candidate formation and capacity growth, not
only committed-slot routing.

Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_online_reliability_delay_promoted_2026-08-10/`.

## Reliability-gated nonlinear candidate growth (2026-08-10)

The learned evidence boundary now reaches candidate admission and verified
capacity growth in a three-seed nonlinear factored audit. A frozen source regime
was promoted, clean verifier outcomes warmed the replay-free reliability state,
and a low-error corrupted revisit was vetoed and quarantined without staging a
replacement candidate. A fresh gate-disabled control accepted the same
corruption. High-error novel nonlinear evidence remained eligible for an
isolated candidate, which promoted while growing the residual capacity from one
slot to two and retaining the original source route.

The base model, controller, and context encoder remained frozen; no old
evidence was replayed; persistence was exact. This closes the boundary where a
reliability veto could otherwise accidentally become a candidate-capacity
policy. It still does not establish unrestricted growth or general continual
learning. The next pressure test is adversarial reliability under repeated
regime reversals and candidate-capacity pressure.

Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_factored_online_reliability_growth_promoted_2026-08-10/`.

## Adversarial reversals under capacity pressure (2026-08-10)

The reliability-to-growth boundary now survives a three-seed adversarial
pressure test. After two nonlinear factored regimes were committed, a
low-error corruption was vetoed without staging a candidate and a fresh
gate-disabled control accepted it. Clean returns to both historical regimes
still routed correctly with the production gate active. A third novel regime
was refused at full two-slot capacity, then admitted only after retention-
verified growth to three slots. All prior slots remained routable after the
new promotion; the base, controller, and context encoder stayed frozen, with
zero old-regime replay and exact persistence.

This closes the current reversal/capacity-pressure gap but remains bounded
continual-memory evidence. The next weakness is repeated growth and eviction
over many cycles, where address stability, compression, and retention must be
shown over a longer lifetime rather than one expansion.

Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_factored_reversal_capacity_promoted_2026-08-10/`.

## Repeated growth and eviction lifetime (2026-08-10)

The factored external-memory lifecycle now survives two independent growth
transactions and two verified middle-slot evictions across three seeds. Seven
nonlinear regimes were promoted; the surviving opaque IDs remained
`(0, 2, 4, 5, 6)` after growth, eviction, new admission, compression, and
restore. Held-out retention passed at each mutation, partial reads did not
mutate the router, and the selected compressed representation passed its
retention round-trip. The base, controller, and context encoder stayed frozen
with zero old-regime replay.

This strengthens bounded memory lifecycle guarantees but does not establish
unrestricted growth or general continual learning. The next pressure test is
to combine repeated lifecycle mutation with online reliability and delayed or
missing evidence, where state admission and eviction must remain safe under
uncertain observations.

Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_factored_repeated_lifecycle_promoted_2026-08-10/`.

## Uncertain-evidence lifecycle (2026-08-10)

The external lifecycle now combines three independently persisted states across
three seeds: factual residual memory, replay-free transition reliability, and
replay-free wait/absence statistics. A low-error corruption was vetoed and
retained in quarantine while active-gate reversals still routed. Delayed
partial evidence was held and resolved once; fast absence released without
mutation. At full capacity a novel regime was refused, then retention-verified
growth admitted it, a middle slot was evicted, and a fourth regime was
promoted. Surviving regimes remained routable and all external state restored
exactly with the base/controller/context encoder frozen.

This is the strongest current bounded uncertain-memory result, but it still
does not establish unrestricted growth, arbitrary computation, or general
continual learning. The next bottleneck is long-horizon stress: many cycles of
uncertain admission, delayed resolution, growth, and eviction with adversarial
reversal and memory corruption controls.

Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_factored_uncertain_lifecycle_promoted_2026-08-10/`.

## Long-horizon uncertain-memory cycling (2026-08-10)

The combined boundary now survives ten nonlinear regimes and four complete
capacity cycles across three seeds. Each run contains four full-capacity
refusals, four retention-verified growth transactions, four evictions, seven
repeated reliability corruption vetoes, seven delayed partial quarantine and
resolution events, and repeated absence no-ops. All regimes promoted, clean
returns routed, final opaque IDs persisted, and reliability/wait/router state
restored exactly while the base, controller, and context encoder stayed frozen.

This is the strongest current bounded uncertain-memory result. It still does
not prove unrestricted growth, arbitrary new computation, or general
continual learning. The next required pressure is adversarial content drift
over these same long horizons: corruption that is deliberately close to a
legitimate transition, verifier reversal, and quarantine saturation.

Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_factored_long_horizon_promoted_2026-08-10/`.

## Near-boundary recovery and quarantine saturation (2026-08-10)

The next pressure test uses deliberately near-tolerance content drift rather
than a clearly corrupted transition. Across three seeds, replay-free verifier
outcomes teach the separate reliability state to veto the drift while the
frozen factored slot remains unchanged. The first two vetoes are retained in
bounded quarantine; when the third arrives at capacity, the route result now
reports `status="reliability_veto"` with
`quarantine_accepted=false`, preserving the distinction between protected
evidence and dropped evidence.

After verifier reversal, the retained bundles resolve and the same drift
routes to the original opaque slot. No candidate is staged, persistence is
exact, and the base and context encoder remain frozen. This promotes explicit
bounded overflow accounting and near-boundary recovery. It does not establish
unrestricted memory growth, arbitrary new computation, or general continual
learning.

Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_factored_boundary_recovery_promoted_2026-08-10/`.

## Context-isolated reliability statistics (2026-08-10)

The factored router now supports replay-free reliability sufficient statistics
addressed by opaque factual context. This closes a real interference hole in
the global error-bin gate: across three seeds, four slots received the same
near-tolerance error but alternating verifier outcomes. The contextual gate
vetoed only the negative slots and preserved positive-slot identity, while a
matched global gate over-vetoed the positive slot.

The contextual state round-tripped exactly, the fact bank remained unchanged
by routing, and the base, controller, and context encoder stayed frozen. This
promotes context-local reliability as a bounded continual-memory primitive;
it does not establish learned raw-modality context formation, unrestricted
memory growth, or general continual learning.

Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_factored_contextual_reliability_promoted_2026-08-10/`.

## Contextual reliability reversal (2026-08-10)

Context-local sufficient statistics now support bounded recency through an
explicit count-decay parameter. Across three seeds, two slots received the
same near-boundary drift while verifier labels alternated twice. The gate
flipped in one subsequent evidence window on each reversal, retained rows
resolved to the correct stable addresses, and the factual bank remained
unchanged. Persistence, frozen-component, and zero-replay controls passed.

This promotes adaptive reliability state, not adaptive factual forgetting: the
factual transition memory remains protected while only the verifier-side
confidence state changes. It does not establish unrestricted memory growth or
general continual learning.

Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_factored_contextual_reversal_promoted_2026-08-10/`.

## Contextual factual memory with goal-conditioned search (2026-08-10)

The current primitives now compose across three seeds: context-local
reliability vetoes near-boundary corruption without mutating factual memory;
verifier reversal releases quarantine; retention-verified growth admits a new
regime; inference-time goal-conditioned search reaches a held-out goal without
an action-policy target; and middle eviction removes the corresponding
contextual reliability state. Exact persistence restores the surviving route.
The controller, shared base, and context encoder remain frozen with zero
replay rows used for reliability calibration.

This promotes a bounded composition of factual external memory and
goal-conditioned behavior synthesis. It does not establish unrestricted memory
growth, arbitrary new computation, learned raw-modality context formation, or
general continual learning.

Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_factored_contextual_search_promoted_2026-08-10/`.

## Disjoint factual-model compounding with no-agent controls (2026-08-10)

The disjoint-dynamics compounding rung now includes an explicit verifier-only
random-intention floor: 128 trials per target and 768 target trials per seed.
Across three seeds, the floor stayed below mastery while matched fresh learners
and policy-free factual-model/search arms mastered both disjoint target
dynamics, retained every prior slot with byte-stable digests, and used zero
old-regime replay during target adaptation.

This strengthens the causal control boundary for disjoint factual-model
compounding. It does not establish unrestricted memory growth, arbitrary new
computation, or general continual learning. The five-seed challenger result
with one rejected cost seed remains the more conservative population audit.

Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_transition_model_disjoint_compounding_controls_promoted_2026-08-10/`.

## Broader disjoint transfer-challenger calibration (2026-08-10)

The transfer-vs-fresh challenger is now tested on seven distinct opaque
dynamics: two sources followed by five targets. Both candidates receive the
same eight-update factual probe and are then trained independently to full
mastery, so probe choice can be compared against actual acquisition cost.
Across three seeds and 15 target comparisons, the probe selected the eventual
lower-cost candidate every time. All candidates mastered, random-intention
floors stayed below mastery, contexts remained separated, prior slots were
byte-stable, and old-regime replay was zero.

This promotes bounded cost-predictive transfer selection. It does not establish
unrestricted memory growth, arbitrary new computation, or general continual
learning; the earlier five-seed negative-cost audit remains a required
conservative control.

Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_transition_model_challenger_calibration_promoted_2026-08-10/`.

## High-noise partial-evidence disjoint routing (2026-08-10)

The learned context router now passes a stronger synthetic noise rung: state
and next-state standard deviation `0.04`, twice the earlier promoted noisy
condition, with target-covering partial evidence and two complete alternation
rounds. Across three seeds, novel regimes were admitted and reused without
labels, all regimes remained mastered, source slots stayed byte-stable,
persistence was exact, and old-slot updates were zero.

This promotes bounded high-noise partial-evidence routing. It does not
establish arbitrary missingness, real multimodal noise, unrestricted memory
growth, or general continual learning.

Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_disjoint_dynamics_noisy_partial_high_noise_promoted_2026-08-10/`.

## Dynamic-regime factual versioning (2026-08-10)

The online context resolver now retains a per-stream history of factual
versions. In a three-seed reversal-cycle pressure test, regime A was admitted
first; two contradictory rows for regime B kept the first row uncommitted and
allocated B at a new address on the second. Subsequent A and B rows reactivated
their existing addresses rather than allocating duplicate versions or writing
over old facts. Both versions remained exact, persistence preserved the route,
the controller remained byte-stable, and optimizer updates and replay were
zero.

The implementation also canonicalizes normalized opaque stream keys before
serialization, preventing a restart from silently creating a duplicate stream
binding due to float32 round-off. This is a bounded same-stream factual
versioning result. It does not establish arbitrary regime discovery, learned
compression, unbounded growth, or general continual learning. The reports and
ledger are archived in
`session_records/sequence_working_memory_2026-08-02/external_online_context_versioning_promoted_2026-08-10/`.

## Copy-on-write after factual-model consolidation (2026-08-10)

The external model bank's verifier-gated parameter sharing now has an explicit
copy-on-write boundary. Across three seeds, two equivalent opaque contexts
shared one physical model after held-out verification, reducing three physical
models to two while preserving both logical addresses. A later adaptation to
the second context detached only that context; the source model remained
byte-stable and all three contexts became independent again. Distinct source
and target functions were rejected without mutation, and persistence preserved
the post-detachment state.

This fixes a dangerous consolidation lifecycle hole: parameter sharing is now
temporary storage compression, not shared plasticity that can overwrite an
unrelated context. The controller and consolidation transaction used zero
optimizer updates. It remains a bounded storage-sharing result, not semantic
merging, unrestricted growth, or general continual learning. Evidence and the
accounting ledger are archived in
`session_records/sequence_working_memory_2026-08-02/external_transition_model_consolidation_cow_promoted_2026-08-10/`.

## Long alternating nonlinear lifecycle after consolidation (2026-08-10)

The nonlinear external-memory lifecycle was extended from a one-shot
compaction check to 16 complete alternation rounds across six regimes. Across
three seeds, the bank grew through capacities `4 -> 6 -> 7`, shared an
equivalent model, detached the later-updated alias copy-on-write, and retained
all six regimes before and after detachment. The selected `float16_stats`
representation preserved the same 96 regime visits per seed after compression
and restore. Historical model digests, opaque logical addresses, router state,
and corruption rejection remained exact. The detachment used 64 fresh
target-regime rows per seed; accounting recorded 896 unique verifier bits and
zero replay. The controller and consolidation transaction used zero optimizer
updates.

This promotes bounded long-alternation storage lifecycle safety. It does not
establish semantic merging, unbounded memory growth, or general continual
learning. Evidence and the accounting ledger are archived in
`session_records/sequence_working_memory_2026-08-02/external_nonlinear_address_compaction_long_cow_promoted_2026-08-10/`.

## Concurrent nonlinear goal-alignment capacity (2026-08-10)

The goal-representation boundary now has a canonical external alignment bank
with stable opaque frontend-space IDs, logical slot IDs, held-out admission,
bounded quarantine, retention-gated growth, and stable-ID eviction. The bank
supports affine and frozen-random-feature nonlinear adapters without putting a
frontend-specific reasoning branch in the controller.

Across seeds `84701`, `84702`, `84703`, and `84704`, two valid alignments
coexisted at capacity two with initial mastery at least `0.9833`. A shuffled
candidate failed its held-out gate and remained quarantined. A valid third
frontend was refused at active capacity, retained in quarantine, and promoted
after eviction of slot `0`; the surviving IDs ended as `(1, 2)` and the
promoted frontend reached at least `0.9833` mastery. Failed eviction left live
state unchanged, persistence was exact, and controller, factual model, and
verifier memory digests stayed byte-stable with zero replay.

This closes concurrent bounded alignment-slot lifecycle safety, not automatic
semantic frontend identity or unrestricted memory growth. The next pressure
test must combine multiple alignments with repeated nonlinear basis growth,
delayed or missing evidence, and learned identity under no privileged space
IDs. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_goal_representation_alignment_bank_promoted_2026-08-10/`.

## Learned opaque identity for concurrent goal alignments (2026-08-10)

Runtime goal-alignment routing no longer requires the caller to provide a
frontend-space ID. The external bank now reuses the generic slot-local route
memory to store bounded prototypes of learned frontend-tensor summaries. A
route is served only when its score and winner margin pass explicit floors;
ambiguous or missing signatures return no aligned output. Identity prototypes
are written only during held-out-verified admission or an explicit
verifier-approved update.

Across seeds `84801`, `84802`, `84803`, and `84804`, runtime routing accuracy
was `1.0` and the shuffled identity control was `0.0`. Ambiguous and missing
signatures were refused in every run. A valid unseen swapped frontend was
quarantined at capacity, admitted after stable-ID eviction, and reached
`0.9667`–`1.0` mastery without a runtime frontend ID. The corrupted candidate
remained quarantined and was not served; persistence was exact, route reads
were non-mutating, the controller/model/verifier stayed byte-stable, and
replay was zero.

This promotes bounded learned signature routing, not semantic open-world
identity discovery. The next gap is learning identity from richer partial
windows under gradual drift and delayed evidence, including cases where two
frontends have overlapping signatures. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_goal_representation_alignment_identity_promoted_2026-08-10/`.

## Delayed identity resolution for overlapping goal alignments (2026-08-10)

The goal-alignment bank now has a bounded quarantine for identity signatures
that cannot yet be assigned safely. Exact signature overlap is refused at the
active route boundary; unresolved evidence can be retained outside active
prototypes, while referenced slots remain protected from eviction. A later
anchor may resolve the evidence only through an explicit verifier-gated
mutation. Rejected anchors preserve the full bank digest; accepted anchors
consume only evidence that successfully updates the selected slot.

Across seeds `84901`, `84902`, `84903`, and `84904`, two overlapping signatures
were retained at quarantine capacity two, a third was refused, eviction was
blocked while evidence was live, verifier rejection was byte-stable, and
verifier acceptance resolved both records. Resolved frontend mastery was
`0.975`–`1.0`; persistence was exact, the controller/model/verifier stayed
frozen, and replay was zero.

This promotes bounded verifier-gated delayed identity resolution. It does not
establish semantic open-world identity discovery, learned anchor selection,
unrestricted memory growth, or general continual learning. The next pressure
test combines this boundary with gradual drift, missing windows, repeated
route reversals, and anchor selection without caller-supplied slot IDs.
Evidence and accounting are archived in
`session_records/sequence_working_memory_2026-08-02/external_goal_representation_alignment_delayed_identity_promoted_2026-08-10/`.

## Caller-free identity under drift and partial evidence (2026-08-10)

Identity routing now accepts an optional learned-evidence mask. Missing
dimensions are excluded from both query and prototype sides of the cosine
comparison, so sparse evidence is not interpreted as a zero-valued identity.
The bank also exposes a verifier-gated anchor operation that selects the
opaque slot itself; callers provide only the verifier's accept/reject outcome,
not a frontend or slot ID. Full accepted anchors may update bounded identity
prototypes and resolve quarantine; partial anchors route but remain read-only
until a complete anchor exists.

Across seeds `85001`, `85002`, `85003`, and `85004`, two alignments survived 16
gradual drift phases with 15 arrival-order reversals per run. Each run routed
all 32 windows correctly, including 24 masked windows, and reached `1.0`
mastery for both affine and nonlinear alignments. Persistence was exact, the
controller/model/verifier stayed frozen, and replay was zero.

This promotes bounded replay-free verifier-gated identity retention under
gradual and reversible drift, partial learned evidence, and caller-free anchor
selection. It does not establish semantic open-world identity discovery,
autonomous verifier design, unrestricted memory growth, or general continual
learning. The next pressure point is a learned masked-prototype or evidence
accumulator that can improve from repeated partial windows without storing
unrecoverable zero-filled identity vectors. Evidence and accounting are
archived in
`session_records/sequence_working_memory_2026-08-02/external_goal_representation_alignment_drift_missing_reversal_promoted_2026-08-10/`.

## Persistent masked identity prototypes (2026-08-10)

The external route-memory boundary now persists an observed-dimension mask
with each verifier-approved identity prototype. Partial queries compare only
dimensions observed by both the query and prototype. Verifier-approved partial
anchors can merge with existing prototypes or append bounded masked variants;
missing dimensions are never represented as learned zero values. Legacy full
prototype payloads remain readable and checksum-compatible.

Across seeds `85001`, `85002`, `85003`, and `85004`, all 33 identity windows
per seed routed correctly, all 25 partial anchors updated persistent masked
memory, and all 33 anchors updated without a caller-supplied frontend or slot
ID. Both alignments reached `1.0` mastery; one masked prototype persisted and
restored exactly in every run. Frozen controller/model/verifier state and zero
replay passed in every run.

This promotes bounded replay-free verifier-gated learning from repeated partial
identity evidence. It does not establish semantic open-world identity,
autonomous verifier design, unbounded prototype growth, or general continual
learning. The next pressure test measures masked-prototype capacity pressure,
verifier-gated consolidation and eviction, and transfer to unseen partial-
window patterns. Evidence and accounting are archived in
`session_records/sequence_working_memory_2026-08-02/external_goal_representation_alignment_masked_prototype_promoted_2026-08-10/`.

## Verifier-gated masked-prototype replacement under capacity (2026-08-10)

Masked identity memory now has an atomic replacement transaction for full
per-slot capacity. A novel partial anchor is staged on a copy; the least-
supported prototype is replaced only if an external retention probe preserves
the existing opaque routes and the new route. A rejected probe leaves the
live digest, masks, counts, and drop telemetry unchanged. The bank exposes
this transaction through caller-free anchor selection, so runtime updates do
not require frontend or slot IDs.

Across seeds `85101`, `85102`, `85103`, and `85104`, unsafe replacements were
rejected without mutation, accepted replacements retained both core routes and
the prior masked route, and the new masked route persisted after reload.
Affine mastery was `1.0`; nonlinear mastery was `0.9917`–`1.0`. The controller,
factual model, and verifier stayed frozen and replay was zero.

This promotes bounded verifier-gated masked-prototype replacement under fixed
capacity. It does not establish autonomous retention policy, unbounded growth,
semantic open-world identity, or general continual learning. The next
pressure test composes replacement with multi-slot masked growth,
verifier-gated consolidation, unseen-mask transfer, and reversal recovery.
Evidence and accounting are archived in
`session_records/sequence_working_memory_2026-08-02/external_goal_representation_alignment_masked_capacity_promoted_2026-08-10/`.

## Persistent masked identity prototypes (2026-08-10)

The external route-memory boundary now persists an observed-dimension mask
with each verifier-approved identity prototype. Partial queries compare only
dimensions observed by both the query and prototype. Verifier-approved partial
anchors can merge with existing prototypes or append bounded masked variants;
missing dimensions are never represented as learned zero values. Legacy full
prototype payloads remain readable and checksum-compatible.

Across seeds `85001`, `85002`, `85003`, and `85004`, all 32 identity windows
per seed routed correctly, all 24 partial anchors updated persistent masked
memory, and all 32 anchors updated without a caller-supplied frontend or slot
ID. Both alignments reached `1.0` mastery; exact persistence, frozen
controller/model/verifier state, and zero replay passed in every run.

This promotes bounded replay-free verifier-gated learning from repeated partial
identity evidence. It does not establish semantic open-world identity,
autonomous verifier design, unbounded prototype growth, or general continual
learning. The next pressure test measures masked-prototype capacity pressure,
verifier-gated consolidation and eviction, and transfer to unseen partial-
window patterns. Evidence and accounting are archived in
`session_records/sequence_working_memory_2026-08-02/external_goal_representation_alignment_masked_prototype_promoted_2026-08-10/`.

## Verifier-gated masked identity-memory growth (2026-08-10)

The external route-memory boundary now supports copy-on-write growth of the
per-slot prototype budget. A capacity increase is committed only after a
retention probe verifies the existing opaque routes on the candidate memory;
rejected growth leaves the live digest and capacity unchanged. Existing rows
are copied directly, so memory capacity can grow independently of controller,
model, adapter, or verifier updates and without replaying old experiences.

Across seeds `85201`, `85202`, `85203`, and `85204`, identity memory grew from
one to three prototypes per slot. Two distinct partial observations with
different masks were then appended for one slot while retaining both original
full routes. Affine mastery was `1.0`; nonlinear mastery was `0.9917`–`1.0`.
Exact persistence, frozen controller/model/verifier state, and zero replay
passed in every run. A strict mask-overlap compatibility gate prevented false
merges when the second mask shared only half of the union of observed
dimensions.

This promotes bounded verifier-gated external-memory growth across changed
partial evidence masks, not autonomous retention policy, unbounded memory,
semantic open-world identity, or general continual learning. The next
pressure point is mask-aware consolidation and capacity policy across more
than two unseen masks. Evidence and accounting are archived in
`session_records/sequence_working_memory_2026-08-02/external_goal_representation_alignment_masked_growth_promoted_2026-08-10/`.

## Bounded masked-memory maintenance stream (2026-08-10)

Route memory now exposes an adapter to the replaceable opaque capacity
planner. The planner sees only normalized prototype keys, learned evidence
masks, generic support, and protection/availability facts; its proposal is
side-effect-free. Verifier-gated growth, replacement, and consolidation remain
the only commit paths.

Across seeds `85401`, `85402`, `85403`, and `85404`, a 28-step stream grew one
slot from two to five rows, learned four differently masked patterns, traversed
forward and reverse order three times, and survived rejected/accepted
replacement and consolidation followed by re-admission and reload. Affine
mastery was `1.0`; nonlinear mastery was `0.9833`–`1.0`; replay-buffer reuse
was zero.

This promotes bounded online maintenance under changed masks and reversal,
with an advisory untrained planner. It does not establish a trained capacity
policy, autonomous retention/compression, unbounded memory, semantic
open-world identity, or general continual learning. The next pressure point is
online planner learning from verifier utility and transfer to longer
interfering streams without replay. Evidence and accounting are archived in
`session_records/sequence_working_memory_2026-08-02/external_goal_representation_alignment_masked_maintenance_stream_promoted_2026-08-10/`.

## Verifier-gated multi-mask identity-memory consolidation (2026-08-10)

External route memory now supports copy-on-write consolidation of selected
slot-local prototypes. Count-weighted masked averaging preserves the union of
observed dimensions; the merged candidate is committed only after a verifier
retention probe confirms every old route. Rejected consolidation leaves the
live rows, masks, counts, and digest unchanged.

Across seeds `85301`, `85302`, `85303`, and `85304`, memory grew from one to
four prototypes per slot, learned three partial patterns with different masks,
then merged two rows while retaining all three partial routes and both full
alignment routes. Affine mastery was `1.0`; nonlinear mastery was `0.9917`–`1.0`.
Exact persistence, frozen controller/model/verifier state, and zero replay
passed in every run.

This promotes bounded verifier-gated external-memory growth and compression,
not autonomous compression policy, unbounded memory, semantic open-world
identity, or general continual learning. The next pressure point is learned
capacity allocation and consolidation over longer streams with interference,
reversal, and bounded storage. Evidence and accounting are archived in
`session_records/sequence_working_memory_2026-08-02/external_goal_representation_alignment_masked_consolidation_promoted_2026-08-10/`.

## Replay-free online verifier-utility learning for capacity planning (2026-08-10)

The opaque capacity planner now supports exploratory masked proposals and a
single-verifier-utility adaptation step. The update changes planner weights
only; it does not mutate memory or the frozen controller. This makes the
capacity policy itself an independently trainable, replaceable component at
the memory boundary.

Across seeds `85501`, `85502`, `85503`, and `85504`, online utility improved
from `0.52`–`0.67` to `1.0` and remained above `0.95` in the measured stable
tail. Deterministic held-out transfer was `1.0` for every trained planner,
versus `0.0`–`0.42` for fresh planners. Each seed used 600 unique verifier
utilities, 600 optimizer updates, zero replay, and a frozen controller.

This promotes online learning of a bounded capacity-maintenance policy for
one redundant-pair consolidation regime. It does not establish a universal
capacity policy, autonomous verifier design, unbounded memory growth, or
general continual learning. The next pressure test must mix admission,
eviction, growth, and consolidation utilities over longer interfering
route-memory streams with reversal controls. Evidence and accounting are
archived in
`session_records/sequence_working_memory_2026-08-02/opaque_capacity_planner_online_utility_promoted_2026-08-10/`.

## Replay-free sequential mixed-action capacity learning (2026-08-10)

The external capacity policy was pressure-tested sequentially: it first
learned consolidation, then learned admission, eviction, consolidation, and
growth from a balanced stream without replaying the pretraining examples.
Verifier utility remained external and scalar; the planner saw only opaque
learned rows plus generic occupancy, protection, and availability facts.

Across seeds `85601`, `85602`, `85603`, and `85604`, mixed online utility rose
from `0.900`–`0.910` to `0.975`–`0.995`. The nontrivial eviction selector rose
from `0.64`–`0.70` to `0.90`–`0.98`. Held-out utility was at least `0.905`
for each action, and the earlier consolidation skill remained at `1.0` after
the mixed phase. Each run used 2,000 unique verifier utilities, 2,000 policy
updates, zero replay, and a frozen controller. The fresh-policy comparison
excluded growth from the gain requirement because all-protected growth is a
deliberately trivial control.

This promotes sequential bounded capacity-policy learning with retention; it
does not establish universal policy composition, autonomous verifier design,
unbounded memory, or general continual learning. The next pressure point is
closed-loop integration with actual route-memory transactions over longer
nonstationary streams, including interference, reversal, and growth cost.
Evidence and accounting are archived in
`session_records/sequence_working_memory_2026-08-02/opaque_capacity_planner_mixed_utility_promoted_2026-08-10/`.

## Closed-loop route-memory capacity-policy transfer (2026-08-10)

The planner is now connected to actual route-memory maintenance transactions.
Verifier-approved replacement can honor the planner's selected row, and the
route-memory maintenance boundary can request exploratory proposals. The pair
selector was upgraded to use coordinate-invariant key/value relations,
generic support/age metadata, and incoming-to-row similarities rather than
raw feature coordinates.

Across seeds `85701`, `85702`, `85703`, and `85704`, copy-on-write
verifier-gated transactions achieved `1.0` held-out utility for admission,
eviction, consolidation, and growth under both forward and reversed
redundancy patterns. The earlier consolidation skill remained at `1.0`;
stable mixed utility was `0.98`–`0.99` or higher. The trained policy's
aggregate gain over a fresh planner on the learnable action families was at
least `0.333` in both patterns. Each run used 2,000 unique verifier
utilities, 2,000 policy updates, zero replay, a frozen controller, and
1,598–1,599 committed transactions out of 1,600.

This promotes closed-loop bounded capacity-policy learning and relational
selector transfer. It does not establish persistent nonstationary memory
management, unbounded growth, universal policy composition, autonomous
verifier design, or general continual learning. The next pressure point is a
single persistent route-memory stream with interference, reversal, and
explicit growth/consolidation costs, evaluated by retention prefixes and
transaction regret. Evidence and accounting are archived in
`session_records/sequence_working_memory_2026-08-02/route_memory_planner_closed_loop_promoted_2026-08-10/`.

## Persistent route-memory stream under interference and growth cost (2026-08-10)

The learned maintenance policy was run against one evolving external memory
rather than independent candidate banks. Mastered anchors, unprotected
distractors, pressure events, and a mid-stream reversal were presented over
300 events per seed. Accepted changes used the same copy-on-write,
verifier-gated route-memory transactions; rejected proposals left the live
store unchanged.

Across seeds `85801`, `85802`, `85803`, and `85804`, trained streams committed
`290`–`297` of 300 transactions, ended with utility `1.0`, grew 72–74 times,
and preserved sampled-prefix and full-final retention at `1.0`. Each run used
300 unique verifier utilities, 300 policy updates, zero replay, and a frozen
controller. Fresh planners committed zero transactions in three controls and
fewer than the trained planner in the fourth.

This promotes persistent bounded memory maintenance under interference,
reversal, and growth cost. It does not establish consolidation selection in
every persistent stream, unrestricted memory growth, universal policy
composition, autonomous verifier design, or general continual learning. The
next pressure point is persistent consolidation opportunity plus explicit
utility costs for growth, eviction, and compression, measured by
retention-adjusted cumulative utility and transaction regret. Evidence and
accounting are archived in
`session_records/sequence_working_memory_2026-08-02/route_memory_persistent_stream_promoted_2026-08-10/`.

## Persistent cost-sensitive route-memory compression (2026-08-10)

The persistent stream now begins with two redundant pairs and repeated
compression opportunities. The verifier assigns utility `1.0` to successful
compression, `0.9`/`0.85` to admission/eviction, and `0.65` to growth, making
capacity cost visible to the independently trainable policy.

Across seeds `85901`, `85902`, `85903`, and `85904`, every stream committed
exactly two persistent compressions, grew 41–48 times, and retained every
mastered route at every measured prefix and at final verification. Stable
utility ended at `0.8125`; trained cumulative utility was `149.8`–`181.9`,
versus `0.0`–`3.8` for fresh controls. Each run used 300 unique utilities,
300 policy updates, zero replay, and a frozen controller.

This promotes persistent bounded cost-sensitive compression with retention.
It does not establish open-ended redundancy discovery, unrestricted growth,
universal economic planning, autonomous verifier design, or general
continual learning. The next pressure point is learned compression discovery
beyond the two seeded pairs, compared against a fixed-capacity non-compressing
controller using retention-adjusted utility. Evidence and accounting are
archived in
`session_records/sequence_working_memory_2026-08-02/route_memory_persistent_compression_promoted_2026-08-10/`.

## Open-world redundancy discovery without seeded duplicate pairs (2026-08-10)

The persistent memory stream now begins with mastered anchors and one
distractor, with zero duplicate pairs preloaded. For each of 50 latent
lifetimes, one route is introduced and a noisy second observation arrives
later. The planner must preserve both, discover the generic relational
similarity, and select verifier-gated consolidation before advancing. A
coordinate reversal occurs after lifetime 25.

Across seeds `86001`, `86002`, `86003`, and `86004`, all 50 lifetimes
completed and all 50 were compressed. Trained runs used 262–691 unique online
verifier utilities; fresh controls completed zero or one lifetime. Sampled
prefix and full-final retention were `1.0` in every run, with a frozen
controller and zero replay. The planner uses a generic key-similarity prior
to make pair exploration tractable; policy actions and selectors still adapt
from verifier utility.

This promotes bounded open-world redundancy discovery and compression. It does
not establish arbitrary semantic equivalence, unrestricted growth, universal
continual learning, or autonomous verifier design. The next pressure point is
false-consolidation control under unrelated high-similarity distractors and
transfer to unseen equivalence transformations. Evidence and accounting are
archived in
`session_records/sequence_working_memory_2026-08-02/route_memory_open_world_compression_promoted_2026-08-10/`.

## Verifier-safe false-consolidation control with unseen evidence patterns (2026-08-10)

The persistent route-memory test now includes an unrelated pair whose raw key
cosine is higher than the true redundant pair. Generic evidence-mask relations
are the only differentiating signal. Copy-on-write retention verification
rejects false pairs and confirms that rejected proposals leave the live store
unchanged.

Across seeds `86101`, `86102`, `86103`, and `86104`, online utility improved
from `0.07`–`0.26` to `0.98`–`1.0`. The trained planner reached `1.0` utility
on two training patterns and an unseen third pattern. It proposed 320 false
consolidations during training and committed none; every rejected proposal was
atomic. Fresh unseen controls scored `0.0`–`0.17`. Each run used 1,200 unique
verifier utilities, 1,200 planner updates, zero replay, and a frozen
controller.

This promotes verifier-safe bounded consolidation control and generic pattern
transfer. It does not establish arbitrary semantic equivalence, learned
verifier design, unrestricted memory growth, or general continual learning.
The next pressure point is delayed, multi-candidate false-consolidation
control over open-ended route identities with reversal and retention-adjusted
utility. Evidence and accounting are archived in
`session_records/sequence_working_memory_2026-08-02/route_memory_false_consolidation_control_promoted_2026-08-10/`.

## Delayed evidence with multiple high-similarity candidates (2026-08-10)

The persistent route-memory stream now presents each identity out of order:
the true first observation, two unrelated but highly similar distractors, and
then a delayed partial observation of the true identity. Only after the delay
does consolidation become available. Copy-on-write verification rejects every
non-target pair, preserves all accepted evidence, and detects a deliberately
corrupted memory state without mutating it further.

Across seeds `86201`, `86202`, `86203`, and `86204`, all 12 cycles completed
through a coordinate reversal. Each run committed 12 compressions and 36
growth operations with minimum prefix and full-final retention `1.0`. The
planner made `95`–`284` false-consolidation proposals per seed and committed
none. Trained planners completed all six unseen-pattern evaluation cycles;
fresh controls completed `0`–`3`. Reward-shuffled controls required
`2,395`–`3,000` attempts versus `198`–`387` for clean training. Every run
used zero replay and a frozen controller.

This promotes delayed verifier-safe bounded capacity maintenance, generic
pattern transfer, and multi-candidate false-consolidation control. It does
not establish arbitrary semantic identity, learned verifier design,
unrestricted memory growth, or general continual learning. The next pressure
point is multiple latent identities in flight with longer delays, cross-
identity distractors, and bounded-memory eviction. Evidence and accounting
are archived in
`session_records/sequence_working_memory_2026-08-02/route_memory_delayed_multicandidate_control_promoted_2026-08-10/`.

## Matched fresh initialization for policy-free transfer (2026-08-10)

The copy-on-write transfer challenger now accepts a caller-owned fresh model.
The probe receives an isolated transfer copy and a probe clone of that fresh
state, while the matched fresh control trains from the exact unprobed digest.
This closes an experimental validity hole: a transfer comparison must not mix
prior quality with random initialization luck. The live source slot remains
byte-stable and only the explicitly selected candidate can enter the bank.

The disjoint four-regime audit was rerun across seeds `70411`–`70415`. All five
seeds mastered both novel regimes, retained every prior slot at `1.0`, kept the
controller frozen, used zero old-regime replay, and passed the cumulative model
cost gate. Warm versus matched-fresh cumulative updates were `155/162`,
`133/145`, `125/131`, `150/157`, and `137/145`; transfer was selected on three
of the ten target decisions.

This promotes a fair, reproducible challenger protocol and a five-seed
policy-free disjoint compounding signal. It does not promote general
continual learning: the dynamics family, context encoder, model capacity,
probe budget, and planner horizon remain finite and synthetic. Evidence is
archived in
`session_records/sequence_working_memory_2026-08-02/external_transition_model_disjoint_compounding_matched_fresh_promoted_2026-08-10/`.

## Shared factual bank with interleaved stream bindings (2026-08-10)

The exported games architecture made the ownership rule explicit: the
controller is fixed-size compute, the external bank is growing factual
memory, and behavior is derived by search rather than stored as one policy per
task. The next implementation consequence is that multiple live streams must
not share one pending evidence window or one provisional candidate.

`ExternalMultiStreamTransitionContextRouter` now provides that boundary. It
keeps pending rows, active continuity, ambiguity quarantine, address-adapter
copies, and provisional factual candidates isolated per opaque normalized
stream key, while sharing exactly one model bank, context encoder, route query,
sparse evidence index, and verifier boundary. Once a candidate is promoted, its
stable slot address becomes that stream's preferred continuation address.
Evidence must still fit the preferred factual model; a contradiction can fall
back to ordinary factual routing rather than silently overwriting the old slot.
This is stream binding, not a task label or a modality-specific reasoning
branch.

The persistence payload stores the shared router once and serializes only
stream-local transient state per binding. Its recursive checksum hashes tensor
dtype, shape, and bytes instead of tensor string representations, so a changed
stream key, candidate, or pending row is rejected before restoration. Bank
context matching must retain the established float32 round-off tolerance;
normalizing an already-stored key again is not bit-exact.

The sub-minute pressure test in
`experiments/external_transition_model_multistream/` stages three streams in
interleaved order, updates only one candidate, verifies the other candidates
remain byte-stable, then performs one-pass factual updates and held-out
promotion. Seed `1901` promoted all three streams, routed revisits to stable
slot IDs `[0, 1, 2, 0, 1, 2]`, restored the same routing after persistence,
rejected a checksum-corrupted payload, used zero replay and zero optimizer
updates, and kept the controller frozen. This promotes a bounded shared-bank
binding invariant only; it does not establish learned stream identity,
unrestricted memory growth, arbitrary computation, or general continual
learning. The next meaningful rung is concurrent streams with missing,
contradictory, and drifting evidence under bounded eviction.

## Concurrent missing, contradictory, and drifting streams (2026-08-10)

The shared stream boundary now has a lifecycle audit rather than only a clean
interleaving audit. With a capacity-two factual bank, stream 0 continued while
stream 1 temporarily lacked evidence; stream 1 retained its own bounded
pending row and later resumed. A contradictory stream-1 bundle returned
`pending` then `conflict` without mutating the committed bank or staging a
replacement. After a retention-verified eviction of stream 1's old slot, the
drifted evidence staged and promoted a new factual version while stream 0's
model digest and binding remained unchanged.

Across seeds `2201` and `2202`, missing isolation, contradiction safety,
retention after drift, exact persistence, and checksum rejection all passed.
The controller remained frozen, optimizer updates were zero, and replay was
zero. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/external_multistream_robustness_promoted_2026-08-10/`.

This promotes bounded robustness and retention-safe replacement, not learned
stream identity. The next architecture bottleneck is to make the binding key
itself emerge from asynchronous learned events and to train delay/reliability
state online, while keeping the factual bank protected from shortcut labels.

The learned anonymous binding follow-up is now promoted under
`session_records/sequence_working_memory_2026-08-02/external_learned_stream_binding_promoted_2026-08-10/`.
Across two seeds, a frozen encoder trained from paired same-stream views
separates three anonymous streams at 100% diagnostic consistency while a fresh
encoder reaches 16.7%. The result also survives a missing arrival and an
interleaving-order permutation, estimates inter-arrival delay, updates trust
from scalar verifier outcomes, and reloads exactly. This qualifies the next
bounded transport boundary only; it does not qualify open-set identity,
general learned delay policy, unrestricted growth, or general continual
learning.

## Outcome-trained lifecycle policy for anonymous binding (2026-08-10)

The replacement boundary now has a replaceable external lifecycle policy:
`ExternalStreamBindingLifecyclePolicy` scores legal pairs of provisional and
live tracks using only opaque prototype vectors and generic observation,
reliability, delay, age, and similarity telemetry. It logs the exact selection
propensity and consumes one scalar verifier outcome per update without replay.
The policy may propose hold, which is essential when all provisional evidence
is contradictory; it cannot commit a replacement by itself.

`ExternalOnlineStreamBindingMemory.replace_verified_track_with_provisional`
performs retirement and admission as one copy-on-write transaction. A rejected
verifier outcome leaves both live and provisional state byte-stable. The
controller and event encoder remain frozen, and the policy is persisted and
checksummed independently so it can grow or be replaced without changing the
controller interface.

The replicated pressure test in
`experiments/external_learned_stream_binding_lifecycle/` uses five anonymous
streams, two live tracks, and three simultaneous provisional identities. Across
seeds `2401` and `2402`, the learned policy achieved `1.0` safe-replacement
accuracy and `1.0` contradiction/hold accuracy. Fresh policies scored `0.125`
and `0.1667`; outcome-shuffled controls scored `0.125` and `0.2083`. Both runs
passed propensity, atomic rejection, exact persistence, frozen-controller,
and frozen-encoder gates with zero replay and zero controller updates.

This promotes a narrow outcome-trained lifecycle proposal mechanism, not a
learned verifier, autonomous eviction economics, unrestricted growth,
arbitrary identity discovery, or general continual learning. The next pressure
point is to couple the learned proposal policy to held-out factual model
retention under real drift and delayed contradiction, then measure whether it
reduces verified experience per newly retained capability.

## Joint learned binding and factual replacement (2026-08-10)

The next boundary is now implemented as one external transaction rather than
two loosely coordinated commits. `ExternalLearnedMultiStreamTransitionContextRouter`
accepts a learned lifecycle proposal plus an independent held-out transition
and stages the binding replacement and factual-bank replacement on isolated
copies. The candidate consumes provisional evidence once through the
`streaming_statistics` factual path, verifies the held-out prediction, and
commits the new anonymous track and factual slot together. The frozen
controller sees neither the raw transition nor the lifecycle action.

The replicated pressure test in
`experiments/external_learned_stream_binding_factual_lifecycle/` exercises five
interleaved anonymous streams: two live, three delayed provisional, one
policy-selected replacement, and one retained sibling. Seeds `2501` and
`2502` both pass learned proposal selection, scalar rejection atomicity,
wrong-held-out rejection, sibling retention, new-slot routing, exact
persistence, frozen-controller/encoder, and factual-bank drift-isolation
gates. The drift control keeps the learned identity matched while the factual
router returns `conflict`; the bank content digest does not change. The run
uses zero replay, zero factual optimizer updates, and zero controller updates.

This is a narrow protocol and memory invariant: learned anonymous binding can
be coupled to held-out factual retention without allowing a failed candidate to
damage existing knowledge. It is not evidence of unrestricted memory growth,
general drift recovery, learned verifier design, or general continual learning.
The next experiment should vary model families and arrival/delay laws and
compare stable retention and transfer against a matched-fresh learner.

## Joint learned binding and factual growth (2026-08-10)

The external memory boundary now supports verified growth as well as bounded
replacement. `ExternalLearnedMultiStreamTransitionContextRouter.grow_with_factual_candidate`
performs stream-capacity growth, factual-bank-capacity growth, provisional-track
admission, and held-out factual promotion on isolated copies. Existing live
tracks and factual slots are digested before the transaction and must remain
byte-stable before the new state can commit. A rejected scalar outcome, a
wrong-held-out stream, or a failed retention probe is a complete no-op.

The promoted audit is
`experiments/external_learned_binding_factual_growth/`, archived under
`session_records/sequence_working_memory_2026-08-02/external_learned_binding_factual_growth_promoted_2026-08-10/`.
Across seeds `2601`, `2602`, and `2603`, it performs two delayed open-set
growth transactions with zero replay, zero factual optimizer updates, and zero
controller updates. Learned proposal selection beats both a fresh policy and a
verifier-outcome-shuffled policy on every seed. A scalar rejection and a wrong
held-out candidate remain atomic; the original factual slot remains unchanged;
both new slots route after promotion; and save/load is checksum-exact.

The factual challenger also makes the external compute boundary explicit: the
affine candidate is retained for the first stream, while the nonlinear second
stream selects the registered random-feature sufficient-statistics candidate
under the tighter held-out verifier. The family is selected from factual
generalization, not from stream identity or a task label.

This promotes bounded replay-free external memory growth and mixed-family
candidate selection. It does not promote unrestricted memory growth, a learned
verifier, arbitrary new computation outside the registered candidate families,
or general continual learning. The next rung is repeated growth over a longer
sequence with stable-retention prefixes, compression/consolidation probes, and
matched-fresh transfer accounting.

## Binding-aware factual consolidation (2026-08-10)

The external memory lifecycle now includes a binding-aware compaction seam:
`ExternalLearnedMultiStreamTransitionContextRouter.consolidate_factual_slots_verified`.
Two opaque stream tracks retain their independent learned keys and stable
factual slot addresses, while equivalent factual model parameters may share
one physical object. Held-out prediction equivalence and a complete retention
probe run on an isolated full-state candidate. A failed candidate, including a
probe that mutates candidate state, is discarded without changing either
binding or factual memory.

The promoted audit is archived under
`session_records/sequence_working_memory_2026-08-02/external_learned_binding_factual_consolidation_promoted_2026-08-10/`.
Across seeds `2701`, `2702`, and `2703`, three anonymous tracks and three slot
addresses remained present while physical model storage fell from three to two.
Unlike model families were rejected atomically; exact persistence, frozen
controller/encoder, zero replay, and zero factual optimizer updates passed.

This promotes bounded retention-safe factual parameter sharing, not semantic
stream merging, learned maintenance policy, unrestricted growth, or general
continual learning. The next integration is a finite-budget lifecycle that can
learn whether to grow, share, compress, or defer while preserving the same
retention floor.

## Learned finite-budget maintenance choice (2026-08-10)

The memory side now exposes a versioned `grow/share/compress/defer` policy
boundary. A frozen controller remains a generic event-to-intention processor;
the replaceable policy sees only generic external-memory telemetry and a
structural action mask. It performs discrete action selection and can adapt
from one scalar verifier utility without replay. It never receives modality
formats, task IDs, semantic slot labels, or protocol actions.

The router connects this proposal boundary to existing verifier-gated factual
growth and binding-aware parameter sharing. Runtime compression also has an
explicit copy-on-write commit: an independently restored candidate must pass
retention and mutation-integrity checks before the live factual bank changes.
Stable opaque slot addresses and binding tracks are therefore preserved while
physical storage can change.

The bounded pressure test in
`experiments/external_memory_maintenance_policy/` beats matched fresh and
reward-shuffled controls on three seeds, with zero controller updates, zero
replay, and one policy update per unique verifier utility. This is the
architecture's first learned finite-budget maintenance rung; it does not yet
claim learned verifier economics, autonomous candidate equivalence discovery,
unrestricted memory growth, or general continual learning.

The follow-on real-transaction audit in
`session_records/sequence_working_memory_2026-08-02/external_memory_real_maintenance_promoted_2026-08-10/`
derives utility from actual growth, held-out factual sharing, compression-byte,
and retention-probe receipts. All three seeds reach `0.95` held-out utility
against fresh controls at `0.70`, `0.7375`, and `0.70`; persistence and
mutating-probe atomicity pass. The action-shuffled controls reach `0.25`,
`0.70`, and `0.2375`, confirming that the causal maintenance choices matter.
The next architectural pressure is a single long nonstationary stream with
accumulating interference and repeated maintenance decisions, not independent
reset scenarios.

## Long nonstationary external-memory maintenance (2026-08-10)

The maintenance contract has now been extended from four actions to a
versioned five-action boundary: `grow`, `share`, `compress`, `evict`, and
`defer`. `evict` is not an unverified delete; it addresses a stable logical
slot and commits only when every mastered slot passes an independent held-out
retention probe on a copy-on-write candidate.

The long-stream audit in
`session_records/sequence_working_memory_2026-08-02/external_memory_long_nonstationary_promoted_2026-08-10/`
keeps one bank alive for `640` unique verifier utilities. Three seeds repeat
real growth, factual sharing, compression, and eviction, acquire four recurring
opaque capabilities, retain all mastered behavior above `0.9991`, and persist
exactly with zero replay and zero controller updates. Trained online utility
is `0.9953`, `0.9969`, and `0.9969`; shuffled-verifier controls reach only
`0.5156`, `0.5328`, and `0.5188`.

This promotes bounded repeated maintenance economics, not unrestricted memory
growth, learned candidate discovery, arbitrary new computation, or general
continual learning. The next pressure is to make candidate identity and
equivalence discovery emerge from partial event streams rather than a
predeclared schedule.

## Policy-free factual execution (2026-08-10)

The architecture now exposes a first-class runtime path for the strongest
continual-learning result from the exported games session. A policy stores
preferences and can become wrong on a novel regime; a factual transition model
stores what an opaque state becomes after an opaque intention. New evidence can
therefore be added or represented as an external residual without overwriting
the action preference used by an earlier regime.

`PolicyFreeAmodalRuntime` keeps the controller as the single amodal cognitive
component for event integration and working state, but does not decode its
direct intention. It maps the controller's learned state representation
through the independently replaceable `ExternalControllerStateAdapter`,
retrieves a factual model when a bank is present, searches toward an opaque
goal/destination, and sends only the planner's next intention to the existing
intention bus. The controller's direct intention remains diagnostic so it can
be compared causally; it is not the deployed action path in this mode.

This realizes the architectural split:

```text
plant/controller: reusable learned state integration and working memory
external memory: factual transition models, residuals, and opaque goals
runtime compute: model-based search; no stored task policy
```

The exported session also supplies two design rules that remain normative:

1. Diversity identifies abstraction. Train shared parameters on structurally
   different regimes so task-specific pursuit habits conflict; store the
   surviving shared structure in the plant and contradictory facts externally.
2. Optimize lifetime acquisition cost, not only immediate reward. Retrieval
   must be attempted before new learning, and the cost of warm-up must be
   amortized across later targets. A flat library is not compounding.

The runtime seam itself is now implemented and unit-tested, but it does not
promote general continual learning. Promotion still requires fresh and
policy-based controls, zero-shot target capability, held-out model
verification, model-corruption controls, search-depth/latency accounting,
and stable retention over a genuinely novel multi-target stream.

## External opaque intention repertoire (2026-08-10)

The policy-free seam no longer requires a caller-authored candidate tensor at
execution time. `ExternalIntentionRepertoire` is an append-only external
memory of observed opaque intention vectors. It deduplicates vectors by a
versioned representation-space threshold, keeps outcome and exact logging
propensity sufficient statistics without replay, and round-trips through a
checksummed payload. The controller and decoder interfaces remain unchanged.

The repertoire is deliberately not a reward-ranked policy. Verified entries
are exposed as a runtime-sized candidate set and factual model search derives
the action sequence for the current opaque goal. A novel controller intention
is marked as ephemeral exploration and is excluded from verified search by
default; it can be enabled explicitly or used as the safe fallback when the
repertoire is empty. The three-seed promoted audit under
`session_records/sequence_working_memory_2026-08-02/policy_free_intention_repertoire_promoted_2026-08-10/`
reaches every held-out goal, beats the matched empty-repertoire learner, and
keeps the controller/model frozen.

This promotes external candidate retrieval and exploration isolation, not
arbitrary intention synthesis, learned verifier design, unrestricted growth,
or general continual learning. The next pressure is to acquire new intention
vectors from partially observed experience and verify them before they enter
the deployed candidate set.

## Verifier-gated intention admission (2026-08-10)

`ExternalIntentionRepertoire.admit_verified` now applies the same
copy-on-write discipline used by factual model growth to output content. A
novel opaque vector is staged on an isolated repertoire copy; a caller-owned
held-out verifier may test it and record its scalar outcome, but every prior
entry must remain byte-equivalent and exactly one new entry may be added.
Rejected or mutating candidates leave the live repertoire unchanged.

The three-seed audit in
`session_records/sequence_working_memory_2026-08-02/policy_free_intention_admission_promoted_2026-08-10/`
shows the causal value of the boundary: a diagonal goal is not mastered with
the retained repertoire, becomes mastered after verified admission of one new
opaque intention, and a mismatched candidate is rejected. The controller and
factual model remain frozen.

This is verified new-intention storage, not a learned generator of arbitrary
output programs. The next pressure is to make the candidate source itself
emerge from partial experience and active exploration while retaining the same
held-out admission and complete-retention gates.

## External compositional intention exploration (2026-08-10)

`ExternalIntentionCompositionExplorer` now supplies a first bounded source of
new candidates from retained opaque experience. It composes pairs of verified
intention vectors with a versioned operation set (`mean`, `sum`, and
`difference`), deduplicates near-equivalent proposals, records source indices
and operations, and never mutates the live repertoire. The resulting proposal
is ephemeral until `admit_verified` accepts it through an independent held-out
verifier.

The promoted admission audit now proves this causal chain: the diagonal
intention is generated from entries `(0, 1)`, admitted, and immediately enables
the held-out goal while the controller, model, and retained entries stay
unchanged. This is a useful pressure-tested bridge from retrieval to
composition, but it is not arbitrary program synthesis, learned operation
discovery, or general continual learning. The next step is to generate and
verify candidates from partial multimodal experience and active outcome-only
exploration rather than a fixed algebraic catalogue.

## Signed external-entry value factorization (2026-08-10)

External memory entries may need to change the polarity of an existing
prediction without forcing the shared state representation to relearn its
salience. `ExternalSignedEntryValueModel` defines this as a versioned,
protocol-agnostic boundary: the state tensor produces a strictly positive
polarity-free salience, the opaque external-entry tensor produces an odd
scalar polarity, and factual value is their product. A zero entry is neutral;
negating an entry negates the value by construction.

The promoted three-seed audit under
`session_records/sequence_working_memory_2026-08-02/signed_entry_value_promoted_2026-08-10/`
trains only on positive entries, freezes the model, and transfers to held-out
negative entries with zero target updates. The boundary is reusable across
modalities because it consumes learned state and entry tensors, not raw
formats or hand-assigned semantics. It is not yet arbitrary value learning,
unrestricted memory growth, or general continual learning. The live
`ExternalModelBasedPlanner` seam now accepts runtime-sized opaque
`candidate_entries` and an explicit entry-value weight, so the external model
can alter searched behavior without changing the controller, transition model,
or decoder protocol. The next pressure test is this search path across
changing external regimes without replay.

## Persistent external entry repertoire (2026-08-10)

`ExternalEntryRepertoire` is the independent long-term store for factual
value entries. It grows append-only, deduplicates near-equivalent opaque
vectors, records outcome/propensity sufficient statistics without replay,
round-trips through a checksummed payload, and admits novel entries only via
an isolated held-out verifier. `PolicyFreeAmodalRuntime` retrieves its
runtime-sized proposal and can record post-search outcomes without updating
the controller or entry-value model.

This establishes the files-like memory lifecycle, not unrestricted growth or
learned compression. `ExternalEntryBindingRepertoire` now stores
intention↔entry pairs atomically, and policy-free runtime proposals return both
tensors from one external record rather than joining two independently ordered
lists. The next pressure is retention-safe consolidation and compression
across changing regimes with stable logical IDs preserved through maintenance
and held-out factual retention as the gate.

## Retention-safe external binding consolidation (2026-08-10)

The files-like memory boundary now supports verifier-gated compaction without
invalidating durable references. `ExternalEntryBindingRepertoire.consolidate_verified`
builds a copy-on-write candidate, aggregates the retired records' outcome and
propensity sufficient statistics without replay, and introduces one opaque
replacement pair. A caller-owned held-out retention probe must accept the
candidate without mutating it before the live repertoire changes.

Stable logical IDs remain addressable after physical consolidation. One retired
ID is retained as the replacement address and the other retired IDs resolve
through checksummed aliases; aliases, record order, statistics, and the next-ID
counter survive payload round-trips. `PolicyFreeAmodalRuntime` exposes the same
operation while keeping maintenance outside the controller and decoder
protocols. Unit coverage verifies successful aggregation, persisted alias
resolution, and atomic rejection of a mutating retention probe.

This promotes a retention-safe external-memory maintenance primitive, not
learned semantic equivalence, autonomous eviction economics, unrestricted
memory growth, or general continual learning. The next pressure is to train
candidate discovery and retention decisions from verifier outcomes over a long
nonstationary stream, then compare stable retention and transfer against a
matched fresh learner.

## Live signed-entry search (2026-08-10)

The external value contract now reaches factual behavior derivation.
`ExternalModelBasedPlanner` accepts a versioned external entry-value model,
runtime-sized `candidate_entries`, and an explicit nonnegative value weight.
For each terminal expansion it evaluates the predicted learned state with the
matching opaque entry and subtracts that factual value from the search score.
The controller, transition model, and decoders are unchanged; bank-backed
model selection forwards the same boundary through each factual slot.

The promoted three-seed audit under
`session_records/sequence_working_memory_2026-08-02/signed_entry_search_promoted_2026-08-10/`
trains only on positive entries, freezes the value model, and reverses the
selected intention when only the external entry assignment reverses. The
matched no-entry planner remains polarity-insensitive. This promotes live
signed-delta search, not arbitrary value learning or general continual
learning. The next pressure is persistent entry growth and changing-regime
search with independent held-out factual verification.

## Stable-address intention memory (2026-08-10)

`ExternalIntentionRepertoire`, the standalone output-candidate memory used by
the policy-free fallback path, now has the same retention-safe lifecycle as
atomic intention↔entry binding memory. Each observed opaque intention receives
a stable logical ID. Consolidation aggregates its outcome and exact-propensity
sufficient statistics on a copy, retains one replacement address, persists
aliases for the retired IDs, and commits only after an independent held-out
retention probe passes without mutation.

Runtime proposals and composition provenance use logical IDs rather than
physical row positions. Old payloads without address metadata load with their
original positional IDs, preserving the versioned external-memory boundary
while enabling future compaction. The controller, planner, and decoders remain
unchanged; maintenance stays outside the cognitive core.

This closes a consistency and address-stability gap, not the harder problem of
inventing useful new intention vectors from partial multimodal experience.
Learned candidate generation, equivalence discovery, unrestricted growth, and
general continual learning remain unqualified.

An opt-in adaptive evidence-version curriculum now addresses the fixed
boundary in the earlier rejection. Each of seven masks must pass stage-local
mastery and a held-out prefix verifier before a protected copy-on-write child
is created. The child copies reusable content and the route key on previously
observed dimensions; the sole unqualified child receives a temporary `0.75`
exploration floor so discovery remains caller-free. With a four-update stage
minimum, three seeds pass all retention, reversal, causal, persistence,
frozen-core, and zero-replay gates, with warm/fresh successor updates of
`39/50`, `42/44`, and `34/55`. This promotes bounded adaptive sequential
reuse, not arbitrary distribution shift or general continual learning. See
`session_records/policy_free_intention_masked_routing_adaptive_promoted_2026-08-10/`.

The external generator now adds an opt-in factorized masked-content boundary.
In `mask_stable_content` mode, observation masks remain explicit evidence for
routing and retention but are disconnected from the mutable nonlinear hidden
content path. `factorized_context_residual` adds a separate learned residual
from observed values plus bias to the opaque intention; it is external state,
is copied on write, receives delayed outcome credit, and is persisted through
generator schema v2. Older compatible generator v1 payloads migrate with zero
residual state. The two-seed overlapping-mask audit promotes this factorization as a
bounded reuse mechanism, with warm/fresh successor update counts of `9/26`
and `11/20`; it does not change the controller boundary or qualify general
continual learning. See
`session_records/policy_free_intention_masked_routing_factorized_promoted_2026-08-10/`.

## Outcome-trained intention content generation (2026-08-10)

`ExternalOutcomeIntentionGenerator` adds the missing provisional-content
boundary without adding a controller or protocol-specific branch. It maps a
learned opaque context through a compact external stochastic neural generator,
samples a continuous intention, and updates only its persisted external state
from scalar verifier outcomes using Gaussian score-function credit. The state
contains the generator cells, delayed eligibility traces, baseline, counters,
and protection mask; the controller remains frozen and receives no raw
modality data or privileged target information.

Generated content is explicitly provisional. A caller must pass it through
`ExternalIntentionRepertoire.admit_verified` before it becomes a durable
candidate for `PolicyFreeAmodalRuntime` and factual model search. Cells can
grow copy-on-write from a protected predecessor, while missing feedback is an
exact no-op for learned content. Proposal log densities are retained so
outcome accounting can distinguish sampled exploration from verified memory.

The focused causal coverage shows outcome-driven movement toward hidden
continuous verifier targets, shuffled-outcome failure, protected retention,
copy-on-write growth, and exact reload. This is a bounded external proposal
mechanism, not arbitrary program induction or general continual learning. The
next promotion must use partial multimodal contexts, delayed and noisy
outcomes, competing retained candidates, a matched fresh learner, and the
required unique-verifier-bit, update, replay, latency, transfer, and stable
retention accounting.

## Canonical policy-free generator integration (2026-08-10)

`PolicyFreeAmodalRuntime` now integrates the external generator as a
memory-side proposal source. The controller still follows the canonical
`event-token window -> controller/memory -> opaque state` path. The generator
consumes only that adapted state and returns a provisional opaque intention;
the runtime may plan with it immediately for a factual probe, or append it
after verified repertoire candidates. Generator state is caller-owned and is
updated only through explicit decision and scalar-feedback methods, so the
inference path cannot silently write durable memory.

The generator is intentionally incompatible with atomic entry-binding
proposals in this API. A future binding path must supply and verify the matching
entry explicitly; this keeps intention invention separate from
intention-entry commitment and prevents a new output vector from smuggling an
unverified protocol artifact into the planner. Held-out admission remains the
only path to durable candidate memory.

The two-seed bounded audit in
`session_records/policy_free_intention_generation_2026-08-10/` passes frozen
controller/state-adapter, copy-on-write retention, fresh-transfer,
shuffled-outcome, exact-persistence, and zero-replay gates. This is the first
promoted end-to-end seam for outcome-only continuous intention discovery, not a
claim of general continual learning, unrestricted growth, or arbitrary new
computation. The next architectural pressure test is a nonstationary stream
with partial multimodal contexts, delayed/noisy outcomes, competing old
candidates, reversal, and repeated append/protect/consolidate cycles.

## Independent external intention-cell capacity (2026-08-10)

The generator boundary now distinguishes controller batch from memory
capacity. `ExternalOutcomeIntentionMemory` queries every external cell from
each learned opaque controller context and returns a runtime-sized candidate
tensor. The factual planner records the selected candidate index alongside its
opaque intention, so delayed scalar feedback can update the exact external
cell even when copied cells have identical initial content. The controller,
event bus, and decoder interfaces are unchanged.

The memory proposal carries its score gradients as ephemeral evidence. This
keeps delayed feedback causal without replaying old examples or consulting
unattempted candidate outcomes. Protected cells receive no parameter update;
their stable content remains available for factual search and retention probes.

The replicated nonstationary audit in
`session_records/policy_free_intention_memory_2026-08-10/` masks partial
context, delays and noises outcomes, performs repeated growth, and exercises
verified repertoire compaction. An inherited reversal candidate is evaluated
transactionally and rejected when it exhibits negative transfer; a fresh
external cell then acquires the reversal while old cells remain byte-stable.
This promotes independent memory capacity and rollback safety, not learned
cell routing, unrestricted growth, or general continual learning. The next
architectural pressure is a learned opaque router that chooses among cells
without a caller-provided lifecycle address.

## Learned opaque external-cell routing (2026-08-10)

The next boundary is implemented by `ExternalOutcomeIntentionRouter`. It keeps
the controller-to-memory interface opaque while adding a memory-side
context-conditioned route distribution over runtime-sized external cells.
Unseen cells receive bounded exploration; the sampled cell emits the only
candidate passed to the factual planner. Routing precedes content generation,
so a sparse proposal materializes only the selected physical cell IDs rather
than the whole bank for a single controller context. A delayed scalar verifier
outcome updates the selected cell's content and the route score using
proposal-specific propensities. The controller is unchanged and never sees
physical cell IDs, raw modality data, or protocol-shaped actions.

The route proposal is independently versioned and serializable. It contains
selected-cell provenance for delayed credit, but that provenance stays on the
external memory side and is not a semantic field in the controller state.
Append, protect, rollback, missing-evidence no-op, and tensor-only reload
remain copy-on-write operations. This preserves the N-encoder -> one
controller/memory -> M-decoder boundary while removing the previous caller
address requirement.

The replicated audit is archived under
`session_records/policy_free_intention_routing_2026-08-10/`. It promotes
caller-free bounded routing and rollback safety only. It does not establish
unrestricted memory growth, compression, arbitrary new computation, or
general continual learning. A fresh cell cloned from the exact pre-source
initial state provides a matched transfer control; both seeds show faster
successor acquisition from the protected source-derived cell. Route-cost
scaling, compression, and long-horizon stable-prefix transfer remain the next
bottlenecks.

## Portable external compute-slot artifacts (2026-08-10)

The CPU/files boundary now persists learned external computation as a first-
class artifact. `ExternalRegisterComputeBasisArtifact` snapshots one
append-only compute slot's versioned ABI and tensor state, computes an
interface-and-content SHA-256 digest, rejects tensor or configuration
corruption, and restores the slot into a compatible interpreter without
serializing the shared controller or changing its parameters.

`ExternalCapabilityRegisterMachine.basis_artifact()` and
`add_basis_artifact()` make the lifecycle explicit: acquire a slot, freeze and
verify it, write the opaque slot file, then load it into a replacement/frozen
interpreter as independently owned computation. The restored slot produces
the same register transition, while the shared interpreter remains
byte-stable and later mutation of the restored copy cannot affect the source.

This closes a real implementation gap in the CPU/files analogy: an
instruction vector was portable before, but a newly learned computation was
still tied to the in-memory register machine. It is a persistence and
replacement improvement, not evidence of arbitrary new computation or general
continual learning. Behavior verification, retention gates, and explicit
instruction/basis routing remain required before a loaded artifact becomes
deployed capability.

## Explicit partial-context contract for external learning (2026-08-10)

The external intention learner now has an opt-in masked-context ABI. A caller
may provide an opaque context together with a boolean observation mask; the
memory-side learner receives `[observed values, observation mask, bias]` while
the controller width and event-token boundary remain unchanged. Dense callers
continue to use the original `[context, bias]` feature layout and existing
generator files remain loadable.

This matters for continual learning because an absent feature must not become a
learned zero. Value credit is zero for unobserved dimensions, while the mask
channel lets the external learner distinguish “not observed” from an observed
zero. The route score uses observed values only. Routed retention state also
keeps per-dimension observation masses, so partial prototypes update only the
dimensions actually seen and survive append, protection, reversal, and tensor-
only reload. Older routed-memory v1 payloads migrate with zero observed mass;
v2--v3 payloads are known dense-context files, so their existing total masses
are expanded across every context dimension to preserve their old behavior.

The focused causal tests cover masked feature construction, missing-value
gradient exclusion, partial retention prototypes, exact reload, and v3
migration; the repository suite remains green at 614 tests. This is a safety
and information-preservation boundary, not a claim of arbitrary missing-stream
reasoning: the next experiment must test partial contexts under delayed/noisy
outcomes, contradictory streams, fresh controls, reversal, and stable-prefix
retention with the required verifier-bit and replay accounting.

`PolicyFreeAmodalRuntime.observe(..., intention_context_mask=...)` forwards the
same mask to whichever external generator, memory, or router is active. This
keeps partial-evidence handling on the replaceable memory side while leaving
the controller's canonical input/output contract unchanged.

## Mask-aware external routing and transfer boundary (2026-08-10)

The mask ABI now reaches the external address layer as well as the intention
generator. In masked mode, a route key addresses
`[observed_context, observation_mask]`; dense mode retains the original route
width and state layout. New masked cells initialize their mask-specific route
weights neutrally, and copy-on-write generator cells transfer the value pathway
while resetting mask-specific input weights. This prevents an inherited source
observation pattern from being silently treated as the successor's identity.

The complementary-mask pressure test in
`session_records/policy_free_intention_masked_routing_2026-08-10/` is rejected
for promotion: frozen-core, explicit-mask, delayed/noisy, causal, corruption,
persistence, and protected-retention gates pass, but only one of two seeds is
faster than the matched-fresh successor. This is evidence for a real remaining
bottleneck—reliable transfer across changing observation patterns—not evidence
against the mask contract. The next rung is overlapping masks followed by a
gradual mask curriculum; arbitrary missing-stream reasoning remains unclaimed.

## Promoted overlapping-mask transfer and non-destructive reversal (2026-08-10)

The next rung closes that bounded transfer gap. The router now uses a generic
verified-support prior: a protected cell is penalized when its verified
prototype covers too little of the query's observed dimensions. This is an
address reliability signal, not a semantic field or task label. Copy-on-write
children also neutralize value and route weights for dimensions the source
cell never observed, preventing untrained missing dimensions from contaminating
the inherited basis.

Masked reversal is now non-destructive. Repeated relevant low outcomes put a
protected cell into quarantine and demote it for routing while leaving its
content and verified prototype unchanged. A challenger cell can acquire the
new behavior, preserving both versions. Weakly overlapping evidence is ignored
for reversal accounting when it does not cover enough of the verified query;
this prevents an incomplete stream from erasing a mastered capability.

The promotion-quality audit in
`session_records/policy_free_intention_masked_routing_overlap_promoted_2026-08-10/`
passes across seeds `85301` and `85302`. Successor transfer required `9/23`
and `11/19` updates against matched fresh learners, for ratios `2.5556` and
`1.7273`; both seeds passed frozen-core, delayed/noisy, causal, corruption,
exact-reload, protected-retention, held-out prefix verification, and zero-
replay gates. The audit qualifies bounded overlapping-mask transfer and
non-destructive masked reversal, not arbitrary missing-stream cognition,
unrestricted growth, compression, or general continual learning. The
halfway-switch gradual curriculum is archived as a rejection in
`session_records/policy_free_intention_masked_routing_gradual_rejected_2026-08-10/`
and the seven-stage mask-drift curriculum is archived in
`session_records/policy_free_intention_masked_routing_multistage_rejected_2026-08-10/`.
Both reject sequential mutation of one nonlinear cell across evidence
distributions. The next architectural boundary is versioned or factored
reusable computation across those distributions.

## Context-versioned external memory boundary (2026-08-10)

The next implementation increment adds a persistent evidence profile to each
masked external cell. The profile is an opaque per-dimension observation EMA,
not a hand-assigned semantic coordinate. It is versioned in routed-memory v5,
survives append, delayed feedback, held-out verification, and tensor-only
reload, and migrates v4 files from their observed-mass support.

Profile compatibility is an optional routing prior. A context-version fork
freezes the superseded cell, copies the reusable content basis, starts a fresh
route address, and exposes only newly observed dimensions to the child. This
separates content transfer from address transfer and prevents the old cell from
drifting as evidence changes. The controller remains frozen; all updates stay
in external memory.

The versioned multi-stage pressure test is deliberately not promoted. Across
seeds `85301` and `85302`, source and successor retention, exact persistence,
frozen-core, corruption, missing-evidence, and shuffled-outcome controls pass,
but warm successor acquisition ties the matched fresh control at `121/121`
updates and one noisy reversal control fails. The complete reports and
accounting are archived under
`session_records/policy_free_intention_masked_routing_versioned_rejected_2026-08-10/`.
This isolates the next bottleneck: stable sample-efficient content transfer
through a sequence of evidence versions, not merely safe storage or routing.
The already-promoted overlapping-mask result remains unchanged because the
new profile prior is opt-in for legacy regimes.

## Verifier-selected external prior admission

External memory is now able to treat a copied file as a hypothesis rather
than an unconditional initialization. `ExternalOutcomeIntentionRouter` can
create isolated transfer and fresh cells, run a caller-supplied bounded
verifier-only probe, and return an auditable
`ExternalRoutedIntentionPriorSelectionReceipt` with both scores, state
digests, and the selected branch. The controller receives no protocol-shaped
branch label; this remains entirely on the replaceable memory side of the
amodal boundary.

This transaction is the implementation of “CPU plus files” rollback safety:
the CPU remains frozen, files can grow or fork, and a bad inherited file is
discarded before deployment. The novel challenger audit rejects blind copying
on all three seeds and preserves the mastered source and the later novel file
through reversal. It establishes safe bounded prior selection, not a claim
that the controller can invent arbitrary new computation or that transfer is
always beneficial.

The selection receipt also supports a v2 cost-aware utility: a caller may
provide nonnegative transfer/fresh deployment costs and a cost weight, and
the router selects on `probe_score - cost_weight * cost` while recording both
raw and adjusted scores. Zero-cost calls retain the v1 behavior. This keeps
budget policy on the external-memory side of the amodal boundary instead of
adding a controller branch.

The sequential admission audit now exercises this boundary repeatedly: three
unseen task families are admitted from the same protected successor, each
using v2 cost-aware receipts, and every earlier file passes a complete-prefix
held-out verifier before the next append. This is bounded lifecycle evidence
for the CPU/files analogy, not arbitrary computation or unrestricted memory.

Source discovery is now also memory-side: before a sequential admission, the
router can select a source from verified cells using learned compatibility
rather than receiving a physical source index from the caller. The source
selection receipt is versioned and auditable, and the controller still sees
only learned event/intentions. This closes the fixed-address leak without
claiming unrestricted source generalization.

## Learned external admission cost

The external admission boundary now includes
`ExternalRoutedIntentionCostModel`. This is a replaceable memory-side learner,
not another controller branch: it estimates normalized transfer and fresh
continuation work from masked opaque context values, the verified source's
coverage, and current bank size. After an admission completes, only the
selected branch is updated by a normalized replay-free sufficient-statistics
step. Its versioned tensor state has an independent checksum and persistence
boundary.

The sequential audit in
`session_records/policy_free_intention_learned_cost_promoted_2026-08-10/`
replicates this contract across three seeds. All source, causal, retention,
corruption, reversal, persistence, frozen-controller, and zero-replay gates
pass. One matched-fresh run chooses `fresh` for a nearby target where the
historical hand schedule expected `transfer`; this is retained as the correct
behavior because verifier-selected adjusted utility, rather than a task label
or caller schedule, is authoritative.

This promotes learned memory-side admission economics and removes one more
caller-owned lifecycle decision. It does not establish that the cost model
improves acquisition broadly: the probe still dominates selection in this
small task-family stream. Broad cost generalization, universal positive
transfer, arbitrary new computation, unrestricted growth, compression, and
general continual learning remain open.

## Canonical external computation runtime seam (2026-08-10)

The `INPUT -> PROCESS -> OUTPUT` cycle now has a direct external-computation
path. `ExternalProgramAmodalRuntime` keeps the amodal controller as the only
central processor while a versioned `ExternalProgramArtifact` is executed by
the replaceable `ExternalCapabilityRegisterMachine`. The machine consumes the
controller's learned state representation plus opaque action/outcome
feedback, never raw modality payloads, device action IDs, task labels, or
verifier-private metadata. Its transient register result is projected into
one intention and sent through the existing `M`-decoder intention bus.

The artifact execution is copy-on-write: the persistent external register is
advanced by observation, while the executed register and positional trace are
returned as a checksummed `ExternalExecutionSnapshot`. Retention verification
can reject the file without mutating the controller or committed external
state. An `ExternalSequenceProgramMemory` source can select among portable
opaque files through a replaceable query adapter; physical program slots stay
outside the controller boundary.

This implements the exported session's strongest CPU/files lesson more
faithfully than a task-specific policy table: durable memory may contain
reusable factual or executable computation, while behavior is produced by a
shared interpreter and verified at deployment. The implementation proves the
seam and its isolation, not learned program synthesis or Turing-complete
continual learning. The next promotion must show outcome-only acquisition of
new program files on held-out working-memory families and retain earlier files
without replay or controller updates.

## Transactional executable-file admission (2026-08-10)

The executable-file boundary now has a memory-side admission transaction in
`ExternalSequenceProgramMemory`. A candidate `ExternalProgramArtifact` is
validated against the interpreter ABI, evaluated from an ordered stream of
deterministic scalar verifier outcomes, and appended only after its stable
prefix clears the configured threshold. A rejected candidate leaves the file
count, parameters, and protected files unchanged. Promoted files can be
protected independently of the controller, and the complete bank—including
opaque router weights, file output schemas, protection state, and checksummed
artifacts—can be reloaded through `payload()` / `from_payload()`.

This is the correct external-memory transaction for “learn while frozen”:
the controller and decoder interfaces do not change, raw episodes are not
replayed by the file store, and a bad candidate cannot damage an older file.
`ExternalProgramAdmissionReceipt` records the candidate digest,
stable-bits-to-threshold, and commit slot without assigning semantic meaning
to any latent coordinate. The API establishes safe executable-file staging and
persistence; it does not establish learned program synthesis, arbitrary new
computation acquisition, unrestricted growth, or general continual learning.
The next causal audit remains outcome-only acquisition of genuinely new files
on held-out Brain Workshop families, comparing staged warm candidates with a
matched fresh-file control and measuring complete-prefix retention.

## Outcome-only executable-file admission and context-separated retention (2026-08-10)

The first causal pressure test of this boundary is now promoted across three
seeds in
`session_records/sequence_working_memory_2026-08-02/external_program_file_admission_promoted_2026-08-10/`.
Two opaque files are mastered, a third candidate is admitted only after an
ordered scalar verifier stream contains a stable suffix of at least `32`
outcomes, and a corrupted candidate is rejected without mutating the bank.
The controller and interpreter remain frozen; the learner sees only opaque
events, sampled choices, exact propensities, and scalar outcomes.

The new target is retained in a separate external route cell rather than
being appended into one global policy. This follows a negative control: the
single-policy append caused the new route to interfere with an older context.
Cell separation keeps both routes independently selectable while preserving
the one-controller boundary and allowing external memory to grow.

Across seeds, warm target exact-sequence accuracy is `0.7917--0.8750`, matched
fresh accuracy is `0.8083--0.8708`, and source retention is
`0.9167--0.9292`. Shuffled-outcome controls are `0.0708--0.1417`; wrong-file
controls are `0.1583--0.3458`. All admission, protection, retention,
selection, persistence, frozen-controller, zero-replay, and zero-controller-
update gates pass.

This promotes bounded verifier-gated admission of a portable executable file
and context-separated external route retention. It does not establish
program synthesis, arbitrary new computation acquisition, unrestricted memory
growth, or general continual learning. The next decisive test is outcome-only
candidate generation on a larger, genuinely non-synthetic Brain Workshop
family stream.

## Outcome-only executable-program candidate search (2026-08-10)

The next executable-memory gap is now closed at a bounded rung. The new
`ExternalProgramCandidateSearch` generates copy-on-write instruction
sequences from an opaque instruction bank using generic replace, insert,
delete, swap, and jitter edits. It updates only aggregate operator reward and
acceptance statistics from scalar verifier outcomes; raw verifier rows and
target programs never enter the search state.

The three-seed audit is archived under
`session_records/sequence_working_memory_2026-08-02/external_program_candidate_search_promoted_2026-08-10/`.
One protected one-instruction source composes a held-out two-instruction
target in `1--13` proposals, while a matched fresh atom fails the same
one-edit/256-proposal budget. All runs retain source and target at `1.0000`,
reject corrupted and shuffled evidence, reload exact search/file state, and
keep both interpreter and controller frozen with zero replay.

This promotes outcome-driven one-edit structural synthesis of one portable
external file. It does not establish multi-step beam search, arbitrary
program induction, Turing-complete learning, unrestricted growth, or general
continual learning. The next decisive implementation is a persistent
multi-step candidate frontier evaluated on a real Brain Workshop family
stream, with provisional hypotheses isolated from protected files.

## Persistent multi-step executable hypothesis frontier (2026-08-10)

The next rung is now implemented as `ExternalProgramHypothesisFrontier`. It
keeps provisional opaque executable files outside durable capability memory,
retains a protected root, and persists parent digests, depths, scalar quality,
candidate digests, and aggregate search statistics. Its default exhaustive
mode enumerates finite replace/insert/delete/swap neighborhoods in breadth-first
order; a stochastic mode remains available for later learned proposal priors.
Neither mode exposes instruction meaning to the controller or stores raw
verifier rows. Only a candidate that clears the independent stable-prefix
verifier reaches `ExternalSequenceProgramMemory.admit_verified_artifact()`.

The three-seed promotion is archived in
`session_records/sequence_working_memory_2026-08-02/external_program_hypothesis_frontier_promoted_2026-08-10/`.
A useful one-atom parent composes a held-out three-atom target in `13--28`
verifier evaluations; a matched random parent requires `50--66`. The source
file remains protected and exact on held-out registers, the target reaches
`1.0000` mastery, corrupted admission is a no-op, frontier and file state
reload exactly, and both interpreter and controller receive zero optimizer
updates. This is the first persistent multi-step external-memory search seam,
not evidence of open-ended program induction, unrestricted memory growth,
Turing-complete acquisition, or general continual learning. The next decisive
test is the same frontier on genuinely rendered Brain Workshop task families,
where the verifier and target composition are not synthetic atom sequences.

## Verifier-gated executable-memory lifecycle (2026-08-10)

The external executable-file boundary now has a complete bounded lifecycle,
not just append-only admission. `ExternalSequenceProgramMemory` assigns stable
opaque logical file IDs at admission and retains them across physical
compaction. Copy-on-write transactions support `evict_verified()` for
unprotected files, `consolidate_verified()` for held-out-equivalent files,
and `compress_verified()` for smaller durable representations. Every operation
has a metadata-only, versioned `ExternalProgramMemoryTransactionReceipt`;
rejected operations expose the unchanged live digest and cannot mutate the
source bank.

Equivalence and retention remain caller-owned verifier probes. The memory
store does not interpret task labels, modalities, protocol actions, or raw
verifier rows. Compression is decompressed and behavior-checked before
commit, and corrupted envelopes are rejected by checksum. Logical IDs are
memory-side bookkeeping and are never sent through the amodal controller or
intention bus.

The three-seed promotion is archived in
`session_records/sequence_working_memory_2026-08-02/external_program_memory_lifecycle_promoted_2026-08-10/`.
It passes protected-eviction no-op, non-equivalent consolidation rejection,
equivalent-file consolidation, stable-ID retention, corruption and
mutating-probe controls, durable float16 compression, exact persistence,
canonical runtime traversal, frozen-controller/interpreter, zero-replay, and
zero-controller-update gates. This promotes a bounded external-memory
lifecycle contract. Learned maintenance-policy selection, unbounded growth,
learned compression, arbitrary new computation, and general continual
learning remain unqualified.

## Learned executable-memory maintenance (2026-08-10)

The generic `ExternalMemoryMaintenancePolicy` is now connected to
`ExternalSequenceProgramMemory` through a narrow adapter. It consumes the same
12-field storage telemetry contract and a memory-owned structural mask, then
chooses `grow`, `share`, `compress`, `evict`, or `defer`. It never receives
file IDs, task labels, modalities, protocol actions, raw verifier rows, or
candidate semantics. The selected operation is still committed only by the
existing verifier-gated copy-on-write transaction.

The three-seed promotion is archived in
`session_records/sequence_working_memory_2026-08-02/external_program_memory_maintenance_promoted_2026-08-10/`.
Held-out utility reaches `1.0000` after replay-free online policy updates,
beating fresh and shuffled-verifier controls on every seed, while the
interpreter/controller remain frozen. This is the intended CPU-plus-files
separation: the controller supplies stable computation and the external
memory policy learns lifecycle economics around it.

The result is bounded learned maintenance over executable files, not learned
compression, autonomous verification, unrestricted growth, arbitrary program
induction, or general continual learning. The next decisive pressure is to
join maintenance with persistent multi-step hypothesis acquisition on real
Brain Workshop family streams and charge retention/storage cost in the utility.

## File-scoped executable working state (2026-08-10)

`ExternalProgramAmodalRuntime` now maintains one recurrent register state per
stable external logical file ID. A routed file switch therefore resumes that
file's own working context instead of inheriting the previous file's temporal
state. Newly admitted files receive state lazily; verified eviction prunes
retired IDs; physical compaction does not change state ownership. The
controller width and intention bus are unchanged.

This closes a subtle but important continual-learning failure mode: a memory
bank could previously preserve file contents while mixing the temporal state
of different capabilities at execution time. The runtime audit covers
alternating routes, state non-interference, verified retirement, and reload of
stable IDs. It is still a bounded runtime integrity result, not evidence of
general program acquisition or general continual learning.

Runtime schema v6 supports mixed batch schedules and makes trajectory-aware
addressing the default. Each row may select a different logical file; the
executor is invoked with a row mask per file, and the runtime merges only the
resulting output rows while retaining the individual execution snapshots. The
default route query uses the post-step controller representation plus masked
mean/max statistics over the current learned event window. A final-state-only
adapter remains an explicit compatibility option. This removes the prior
requirement to partition a multi-family batch before the controller can run it
and preserves more evidence for route identification.

The output also exposes the opaque route query and soft per-file probabilities
to the host-side learning boundary. These values are suitable for exact
propensity accounting and delayed scalar route credit; they are not fed back
into the controller, and the controller still never receives physical or
logical file identity.

Route exploration is opt-in. With the default greedy behavior, the runtime
reports propensity one; with an exploration rate, it samples the epsilon-mixture
behavior distribution and reports the selected probability per row. Existing
custom route policies remain authoritative and their returned weights are
normalized as the available propensity surface.

The optional `ExternalOutcomeProgramRouter` is now integrated directly at this
boundary. It receives only the detached opaque route query, applies the
previous tick's scalar outcome before the next selection, records exact route
propensity, and persists its eligibility/policy state inside the runtime
checkpoint. `activate_program()` exposes one append-only admission step for a
new executable file; controller width, interpreter parameters, and existing
file states remain unchanged. This is outcome-driven route adaptation over
admitted files, not learned program synthesis or arbitrary new computation.
If the executable bank is evicted or compacted without a corresponding route
policy migration, the runtime now fails closed instead of reinterpreting a
physical action index as another logical file.

## Durable controller-plus-file working state (2026-08-10)

The runtime state boundary now has a versioned tensor-only checkpoint via
`ExternalProgramRuntimeState.payload()` and `from_payload()`. It persists the
controller's event window, workspace, hidden state, source trust, and growth
registers together with the isolated recurrent state for every stable
external logical file. Restoring the checkpoint does not load executable files
or controller parameters; those remain independently versioned resources.

An exact-resume test covers a mixed-file batch, and an unknown-schema control is
rejected before execution; a deterministic envelope checksum also rejects
tensor corruption. This closes restart loss of active working context,
but it is still persistence infrastructure: it does not prove that the
controller acquires arbitrary procedures or retains them under unrestricted
nonstationary learning.

## Opaque goal-fragment memory and compositional destinations (2026-08-11)

The policy-free factual boundary now includes the destination side of the
CPU-plus-files split. `ExternalGoalFragmentMemory` is an independently
versioned store of opaque target fragments. A fragment is a learned/verified
state vector plus an opaque boolean applicability mask; the controller and
planner never receive a semantic field name, task ID, or memory address.

`ExternalGoalFragmentSet` supports runtime-sized `union` and `intersection`
composition. Union searches for a state satisfying any fragment. Intersection
searches for a state satisfying every fragment by scoring the worst masked
fragment. The resulting destination is consumed by `ExternalModelBasedPlanner`
and `PolicyFreeAmodalRuntime`; behavior remains derived from factual transition
rollouts rather than stored as a task policy. Model-bank selection forwards the
same composed destination to every candidate factual model.

Admission is copy-on-write and verifier-owned. A failed retention probe leaves
the live fragment store byte-stable, while accepted fragments checksum and
reload independently of controller parameters. This makes the “replace only
the differing puzzle piece” idea a real interface rather than a metaphor, but
it does not yet claim learned goal discovery, open-ended fragment induction,
or general continual learning.

## Causal trajectory routing and the transfer objective (2026-08-11)

The exported games work identified a concrete information boundary: a final
state can be identical for two regimes whose histories imply different
continuations. The canonical route-query seam therefore supports two explicit
external statistics contracts. The compatibility contract is
`masked_mean_and_max_v1`; the causal contract is
`recency_weighted_and_latest_v1`, which adds a recency-weighted learned-token
summary and the actual latest retained token without resizing the controller,
planner, or intention bus.

“Latest” is defined by the retained-position mask, not by the count of present
tokens. This matters when evidence is sparse or a window contains a gap: route
identity must follow the event that is actually latest, not an assumed packed
layout. Both contracts remain replaceable memory-side addressing mechanisms;
they do not add modality branches or expose memory identity to the controller.

The export also makes the long-term capability objective operational rather
than rhetorical:

```text
after learning A, acquire novel B with lower verified lifetime cost than a
matched fresh learner, while retaining A and surviving reversal controls
```

That objective requires retrieval before adaptation, factual model-plus-search
behavior rather than copied action policy, structural diversity that forces
shared abstractions to be identifiable, and targets difficult enough that a
fresh learner has measurable acquisition cost. A cheap target can hide both
positive and negative transfer. The canonical ledger must therefore keep
unique verifier bits, logical lifetimes, optimizer updates, replay, latency,
and stable-prefix retention separate, and must retain the fresh, shuffled,
missing-evidence, corruption, and reversal controls. The new causal route
contract improves information preservation, but it is not itself evidence of
general continual learning or universal positive transfer.

The factual-bank prior challenger now has the same explicit cost boundary. A
caller may supply opaque transfer and fresh acquisition-cost estimates plus a
nonnegative cost weight; the copy-on-write probe selects the lower
`prediction_error + cost_weight * acquisition_cost` candidate and records both
raw and adjusted errors in a v2 receipt. With all costs zero, the historical
error-only behavior and v1 receipt remain unchanged. This turns “retrieve
before adapting” into a reversible, auditable decision rule rather than an
unmeasured preference.

The online transition router can now replace those caller-supplied estimates
with a shared `ExternalRoutedIntentionCostLedger`. The ledger is external
mutable memory: it predicts transfer/fresh continuation cost from masked opaque
candidate context, verified source coverage, and bank size; after a candidate
passes held-out and retention verification, the caller may submit one
normalized completed cost and update only the branch that was selected. The
ledger is shared across stream-local routers, checkpointed with its own
versioned payload, and excluded from the controller and factual model bank.
Rejected candidates do not update it. This is a learned acquisition-policy
contract, not evidence of broad cost prediction or general continual learning.

## Goal-conditioned downstream planning boundary (2026-08-11)

The canonical Brain Workshop harness now exercises the missing composition
step after goal-fragment admission. The controller, frontend, and decoder are
frozen; an affine external factual slot consumes three rendered lifetimes once
through sufficient statistics. A state from a held-out learned trajectory is
staged as an opaque goal fragment using a deterministic goal-distance probe,
then admitted copy-on-write and supplied to `ExternalModelBasedPlanner`.

On seed `93`, the learned factual slot planned two steps to the admitted
destination with terminal error `0.00360`, compared with `0.04376` for a matched
fresh slot using the same destination file and candidate intention set. The run
consumed `18` transition rows and `16` unique rendered verifier bits, replayed
no examples,
performed no optimizer updates, and left the controller byte-stable. Missing
evidence and a corrupted candidate were rejected. The result qualifies only
the downstream CPU/files composition: it does not establish multi-step goal
discovery, end-task Brain Workshop mastery, arbitrary new computation,
unrestricted growth, or general continual learning. The three-seed replication
(`91`, `92`, `93`) passed the same bounded gates; the ledger is
`session_records/goal_conditioned_planning_2026-08-11/sample_efficiency_ledger.json`.

## Nonstationary source-retaining goal acquisition (2026-08-11)

The destination composition now has a first nonstationary pressure test. A
source rendered family (n-back-2, opaque cue) is learned first, then a
structurally different target family (n-back-3, different opaque cue) is
learned in a separate replay-free factual slot. A held-out target state is
admitted as an opaque goal fragment and used by two-step model-based search;
the same goal file and candidate intention set are evaluated with a matched
fresh target bank. Source recursive error is checked before and after target
acquisition.

Across seeds `91`, `92`, and `93`, the source slot was byte-stable in `3/3`
runs and the trained target beat fresh in `3/3`; all goal files were admitted
and used. The run consumed `108` transition rows once and replayed no examples
or optimizer updates. This qualifies bounded source retention plus target
goal-conditioned acquisition, not general continual learning, universal
positive transfer, arbitrary computation, or unrestricted growth. Evidence is
in `session_records/nonstationary_goal_conditioned_planning_2026-08-11/sample_efficiency_ledger.json`.

## Online discovered goal-conditioned target (2026-08-11)

The online transition-discovery path now optionally continues through the
destination boundary. The router receives no target cue or task label: it
discovers the target context from opaque transition rows, stages an external
factual slot, and commits it only after multi-lifetime recursive, fresh-
challenger, and source-retention gates. After promotion, the same discovered
slot admits an opaque two-step goal fragment and derives behavior through
model-based search.

With the causal recency/latest state contract and a committed-slot write
firewall and a cost-aware transfer/fresh challenger, `9/24` seeds completed
every gate. All nine recovered the route, used the target goal file, beat their
matched fresh goal planner, and retained the source slot; rejected seeds
remained rejected. The runs consumed `504`
transition rows once, replayed no examples, and performed no optimizer updates.
This is a bounded online-discovery result, not general continual learning or
universal warm-over-fresh transfer. The complete ledger is
`session_records/online_goal_conditioned_discovery_2026-08-11/sample_efficiency_ledger.json`.

An opt-in three-seed learned-cost smoke reached the complete gate on `1/3`
seeds. The other two were rejected before any cost observation, so this does
not promote learned acquisition economics; it identifies stable target
discovery and promotion as the next architectural pressure.

## Aggregate retention verification and one-pass context adaptation (2026-08-11)

The online discovery verifier now treats independent recursive holdouts as a
small evidence set rather than a fail-fast sequence. A staged candidate must
remain below the absolute recursive error bound on every holdout, beat a
matched fresh challenger on a majority of holdouts, and have lower mean error
than that challenger. This preserves a hard catastrophic-regression guard
while reducing sensitivity to one noisy rollout. Under the unchanged
recency/latest, cost-aware, frozen-controller configuration, the complete
24-seed result improved from `9/24` to `10/24`; source retention remained
`24/24`, with `624` unique verifier bits, `504` transition rows consumed once,
zero replay, zero optimizer updates, and an unchanged controller. The complete
ledger is
`session_records/online_goal_conditioned_discovery_aggregate_retention_2026-08-11/sample_efficiency_ledger.json`.

The external context encoder now has an explicit copy-on-write contrastive
adaptation boundary. It consumes a fresh paired opaque-view batch, performs
exactly one optimizer update, retains no observations, and returns a candidate
encoder whose digest can be verified before any route key is changed. This is
the correct contract for future online address learning while committed keys
remain immutable. The first one-pass pressure test did not improve discovery
pass rate, so it is retained as an interface seam rather than promoted as a
capability gain. Random-feature widths `128`, `256`, `512`, and `1024` also
left the failing-seed outcomes unchanged; the active bottleneck is coherent
staged evidence and held-out model fit, not external model width. The result
remains bounded online acquisition, not general continual learning.

## Separating novelty routing from promotion fit (2026-08-11)

The online router now uses separate thresholds for two different decisions.
Committed-slot matching uses a tighter `0.02` factual prediction tolerance so a
novel stream is staged before a source slot can absorb its early evidence;
promotion retains the original `0.05` held-out prediction threshold, recursive
error bound, fresh challenger, and source-retention gates. This is a routing
boundary change, not a verification relaxation.

On the same cost-aware recency/latest 24-seed audit, complete discovery,
promotion, route recovery, target-goal admission/use, and fresh-goal improvement
rose from `10/24` to `14/24`. Source retention stayed `24/24`; the run consumed
`690` unique verifier bits and `576` transition rows once, with zero replay,
zero optimizer updates, and a byte-stable controller. A tolerance sweep from
`0.02` to `0.005` remained at `14/24`, supporting a stable novelty-routing
effect rather than a knife-edge threshold. The full ledger is
`session_records/online_goal_conditioned_discovery_routing_threshold_2026-08-11/sample_efficiency_ledger.json`.

This promotes a better evidence-acquisition boundary, not general continual
learning. Ten runs still do not pass, and the remaining failures are held-out
model-fit, recursive candidate-stability, or retention failures. The next
pressure test must improve the staged candidate's factual fit without replaying
rows or changing the promotion gates.

Two direct controls narrow that next design. Preserving the complete temporal
window summary in a naive `5x` external state made `0/24` runs pass because the
one-pass candidate model became too large for the available evidence. Replacing
the sufficient-statistics candidates with a fresh nonlinear streaming optimizer
also produced `0` complete passes on its diagnostic subset. The existing
factored residual-memory components are therefore the next target: freeze
source computation, learn only a compact context-local residual in external
memory, and verify it under the unchanged held-out and retention gates. These
negative controls are archived in
`session_records/online_goal_conditioned_discovery_representation_pressure_2026-08-11/sample_efficiency_ledger.json`.

## Replaceable factored base for replay-free residual acquisition (2026-08-11)

The factored transition model now treats its shared source computation as an
independently replaceable external model. `ExternalFactoredTransitionModel`
accepts a versioned base implementing the opaque transition interface and
persists its complete `state_payload` alongside the residual memory. The
supported checkpoint registry includes the legacy nonlinear transition model,
affine sufficient statistics, and frozen random-feature sufficient statistics;
unknown base schemas are rejected rather than reconstructed implicitly. This
removes the previous persistence seam where every factored checkpoint silently
reconstructed the legacy MLP base.

The canonical pressure audit is
`experiments/brainworkshop_canonical/factored_residual_base_pressure.py`. It
learns a rendered n-back-2 source regime into an affine external base, freezes
that base and the controller, then stages and promotes an opaque n-back-3
target residual through a random-feature sufficient-statistics backend. Across
seeds `91`, `92`, `93`, `95`, `99`, `100`, `101`, `102`, `103`, all `9/9`
runs staged and promoted the target, beat the matched frozen-base challenger,
retained the source recursive behavior, round-tripped the versioned base, and
left the controller unchanged. The run consumed `189` unique verifier bits
and `216` transition rows once, with zero replay and zero optimizer updates.
The prior nonlinear-base prototype passed `4/9` on the same representative
seed set; the replaceable affine base passed `9/9`.

This promotes a better factored external-memory foundation, not general
continual learning. The remaining pressure is to repeat the result across
multiple target regimes and longer horizons, then test whether an external
residual can acquire genuinely new computation rather than only correcting a
shared transition basis. Evidence and accounting are in
`session_records/factored_residual_base_pressure_2026-08-11/sample_efficiency_ledger.json`.

## Cumulative partial routing and long-horizon sequence pressure (2026-08-11)

The factored router now exposes `route_partial_sequence()`. It preserves the
boundaries of one caller-owned stream, accumulates later partial bundles in a
read-only view, and requires two cumulative evidence bundles by default before
accepting a route. A slot identity that flips as evidence grows is refused.
An unresolved near-tie remains `ambiguous`; the method never writes a route,
stages a candidate, or changes model state. This closes a missing-evidence API
gap without weakening ambiguity refusal. The sufficient-statistics residual
families also expose analytic copy-on-write ridge reparameterization, allowing
a verifier to select regularization for a new slot without replaying any
transition rows.

The longer pressure audit is
`experiments/brainworkshop_canonical/factored_residual_sequence_pressure.py`.
With ten-step lifetimes, three sequential n-back regimes, two independent
holdouts per regime, and ridge candidates `0.001`, `0.01`, `0.1`, `1`, and `10`,
the complete gate passed `0/3` seeds. Eight of nine target slots promoted and
every promoted prefix retained its earlier source behavior and beat its fresh
challenger. The failure was informative: one seed could not stage its third
regime, and the missing-evidence route correctly refused a close factual
near-tie. Reversal and checksum-corruption controls passed on every fully
promoted run; the controller stayed unchanged, with zero replay and zero
optimizer updates.

This is a rejected long-horizon gate, not a capability promotion. The next
architecture target is learned opaque identity under close alternatives:
prefix evidence should improve route confidence while retaining safe
ambiguity refusal. Evidence is in
`session_records/factored_residual_sequence_pressure_2026-08-11/sample_efficiency_ledger.json`.

## Stable identity confirmation (2026-08-11)

`route_partial_sequence()` now has an explicit opt-in
`stable_identity_confirmation` policy. It evaluates each cumulative prefix
with the factual model, allows a close per-prefix margin, and accepts only
when the same factual slot wins across all confirmation bundles. A slot flip,
contradiction, reliability veto, or missing winner remains an explicit
refusal. The path is read-only and does not rewrite route keys, contexts, or
the model.

The prefix address update was hardened at the same boundary: full-view keys
are fixed targets during copy-on-write alignment, and learned query adapters
replace the adapter view while preserving historical opaque route keys. This
implements the export's “bind once, then iterate fixed” principle without
making the address learner authoritative.

The matched six-seed diagnostic recovered `2/6` complete gates with stable
confirmation versus `1/6` baseline; the fresh-seed replication was tied
(`1/3` versus `1/3`). This is a seed-sensitive verifier signal, not a
promoted capability gain or evidence of general continual learning. Keep the
policy opt-in until it improves a held-out learning curve and survives fresh
replication. Reports are in
`session_records/factored_stable_identity_confirmation_2026-08-11/`.

## Copy-on-write prefix address learning (2026-08-11)

The external context boundary now supports a versioned one-pass
prefix-to-full alignment update. It returns an isolated candidate encoder or
address adapter, retains no observations or optimizer state, and leaves
historical keys and the factual transition model unchanged until explicit
caller-owned verification. The factored router can optionally persist these
addresses through a proposal-only route query. A learned address may resolve
only a close factual tie; it cannot override a decisive factual prediction or
force an ambiguous route.

The same router now exposes a read-only disambiguation-probe primitive. Given
plausible opaque slots and candidate intentions, it selects the intention with
the largest predicted next-state disagreement. The caller must execute that
intention and submit its fresh observed consequence through the ordinary
factual and promotion gates; the probe itself never writes memory or chooses a
device protocol. This adds an active-evidence path without turning the
controller into a hand-written solver. The corresponding runtime boundary is
AmodalControllerRuntime.decode_intention() and its policy-free wrapper: it
decodes a caller-owned opaque intention without running the controller again.
The causal unit pressure test demonstrates the intended loop: a synthetic
factual tie is refused, the probe selects the intention whose predictions
disagree, and the fresh opaque consequence resolves the correct slot.

The corresponding pressure arm uses fresh post-promotion evidence and counts
the address optimizer update and additional lifetimes separately. Seeds 91, 92,
and 93 remained 0/3 complete at both learning rates 0.003 and 0.03; the
factual model stayed byte-identical, but partial identity recovery did not
improve. This is a rejected identity control, not a learned capability claim.
The result narrows the bottleneck: the system needs an evidence-efficient
identity mechanism that can recognize when a partial observation is
insufficient before a factual model becomes confidently wrong. The ledger is
`session_records/factored_residual_prefix_address_pressure_2026-08-11/sample_efficiency_ledger.json`.

## Active opaque disambiguation on fresh verifier evidence (2026-08-11)

The active-probe boundary now discounts raw model disagreement by predictive
leverage. Affine and frozen-random-feature external slots already retain
normal-equation sufficient statistics, so
`phi.T @ A^-1 @ phi` provides a support signal without retaining observations.
Unsupported extrapolations are therefore less likely to become diagnostic
intentions merely because their predictions are far apart. This remains a
memory-side selection heuristic: it does not inspect the decoder protocol,
choose a key, or mutate the controller.

The canonical fresh-evidence audit is
`experiments/brainworkshop_canonical/factored_active_disambiguation_pressure.py`.
It trains two frozen-controller residual regimes using rendered Brain
Workshop verifier streams, creates a factual near-tie immediately before an
eligible outcome, requests an opaque intention, executes it through
`AmodalControllerRuntime.decode_intention()` and the ordinary keypress decoder,
and routes the resulting successor read-only. On seeds `41`, `42`, and `43`,
the active probe recovered the target slot on `2/3` fresh lifetimes while the
low-disagreement passive control recovered `1/3`; all six probe requests and
decodes were router/controller-write-free, with zero replay and zero
optimizer updates. Seed `43` remains an explicit failure, so this is a
mechanistic active-evidence signal, not general continual learning or a claim
that the frozen decoder has learned n-back.

The failure is useful: disagreement selection can still be miscalibrated when
the selected probe's actual successor lies outside the learned model's useful
support. A fixed opaque probe-sequence contract and a receding-horizon
reselection arm were then tested for two fresh steps; the active arm resolved
`0/3` targets versus `1/3` for the passive control. The extra step accumulated
model mismatch rather than producing reliable identity evidence, so the
sequence arm is explicitly rejected. Accounting is in
`session_records/factored_active_disambiguation_pressure_2026-08-11/sample_efficiency_ledger.json`.
The rejected sequence ledger is
`session_records/factored_active_probe_sequence_pressure_2026-08-11/sample_efficiency_ledger.json`.

## External opaque intention memory in the canonical agent (2026-08-11)

The canonical Brain Workshop agent now exposes `ExternalIntentionRepertoire`
as an independent memory object rather than hiding candidate vectors in the
controller or decoder. `observe_intention()` records only opaque output
vectors, exact logging propensities, timestamps, and scalar verifier outcomes;
`intention_state_payload()` / `load_intention_state_payload()` persist and
restore that memory without touching the neural `state_dict`. Batched delayed
or absent outcomes use an explicit mask, so missing evidence is not silently
converted into a negative label. The rollout path can opt into these writes
with `record_intention_memory=True` while the controller remains frozen.

This closes an important CPU-plus-files boundary, but it does not invent new
computation. A fresh one-step active-probe arm sourced its candidates from the
verified repertoire and recovered `2/3` targets versus `1/3` for the passive
low-disagreement control—the same result as the earlier observed-intention
pool. The two-step arm remained `0/3` versus `1/3`. The memory boundary is
therefore retained for isolation, persistence, and correct missing-evidence
accounting, but is not promoted as an active capability gain. The next
bottleneck remains calibrated supported intention coverage and factual model
uncertainty. The complete accounting is in
`session_records/factored_active_intention_repertoire_pressure_2026-08-11/sample_efficiency_ledger.json`.

## Slot-conditioned support calibration (2026-08-11)

The external probe boundary now has a versioned
`ExternalTransitionSupportStatistics` memory. It stores only replay-free
Beta-style sufficient statistics over opaque transition-model slot IDs and
predictive-leverage bins. It can be serialized with a checksum, grows until
its caller-owned opaque-slot capacity is reached, and remains read-only during
probe selection. The selector conservatively multiplies the established
leverage prior by the support posterior; sparse calibration cannot erase the
existing extrapolation penalty. No controller, decoder, raw modality, or
device protocol crosses this boundary.

The fresh Brain Workshop audit calibrated the two candidate slots from two
additional rendered lifetimes per seed. Active recovery remained `2/3` versus
`1/3` for the passive control, exactly matching the prior active-evidence
signal, but cost 600 verifier bits and 90 logical lifetimes versus 558 and 84
before calibration. This is therefore a rejected selector promotion: the
memory contract is retained as a clean architectural seam, while the current
leverage-conditioned support rule is not considered a learned capability
gain. The next useful experiment must learn outcome-conditioned diagnostic
utility or coverage that discriminates candidate intentions; adding more
support bins or probe horizon alone is not justified. Evidence is in
`session_records/factored_active_support_calibration_pressure_2026-08-11/sample_efficiency_ledger.json`.

## Outcome-conditioned diagnostic utility memory (2026-08-11)

The probe boundary now supports an independent
`ExternalTransitionProbeUtilityMemory`. It consumes only a fresh scalar
resolution-quality outcome and keeps opaque candidate vectors with
sufficient statistics; missing outcomes remain missing. Its neutral prior
means unknown candidates do not alter the established disagreement and
leverage selector. The memory is serializable with a checksum and never
touches controller, decoder, or factual-model state.

The fresh audit used a route-resolution margin rather than a target-slot
label. Active recovery remained `2/3` versus `1/3` for the passive control,
matching the support-only baseline, but exposure increased to 630 verifier
bits and 105 logical lifetimes. The global intention address is therefore
not promoted as a capability gain. The result localizes the next bottleneck:
diagnostic utility depends on the current factual ambiguity and model
coverage, so the memory key must become context- or probe-profile-conditioned.
Evidence is in
`session_records/factored_active_probe_utility_pressure_2026-08-11/sample_efficiency_ledger.json`.

## Profile-conditioned diagnostic utility (2026-08-11)

The diagnostic utility boundary now accepts an opaque probe profile rather than
keying utility by intention alone. The profile concatenates the candidate
intention with model-derived predictive leverage, disagreement, and support
posterior. It remains an external scalar memory: no target slot, modality,
protocol, controller state, or raw verifier stream is stored. Utility evidence
is confidence-gated, so unknown or sparsely observed profiles preserve the
proven baseline selector instead of allowing a noisy estimate to override it.

The fresh rendered Brain Workshop pressure test kept the controller frozen and
used one verifier outcome per candidate calibration. Active recovery remained
`2/3` versus `1/3` for the passive control, exactly matching the established
support-only and global-utility results, while consuming 630 verifier bits,
105 logical lifetimes, and 723 transition rows. The profile-conditioned
boundary is retained because it is the correct isolation and information
contract, but it is not promoted as a capability gain. The next bottleneck is
not another selector feature: it is repeated, transferable evidence across
uncertainty contexts so that utility can be estimated reliably without replay.
Evidence is in
`session_records/factored_active_probe_profile_utility_pressure_2026-08-11/sample_efficiency_ledger.json`.

## Context-transfer diagnostic utility (2026-08-11)

The external utility boundary now has a second, explicitly separate memory
implementation: `ExternalTransitionProbeContextualUtilityMemory`. It factors
each address into an opaque intention and an opaque uncertainty-context vector,
then uses a bounded cosine kernel to transfer only scalar resolution outcomes
between related contexts. It stores no task identity, target slot, protocol
action, raw verifier stream, or replay trajectory. Effective evidence counts
control confidence, and unrelated intentions or distant contexts remain at the
neutral prior.

The causal unit test confirms the intended behavior, including persistence and
missing-outcome handling. The matched-exposure fresh rendered Brain Workshop
arm used four calibration outcomes per candidate: active recovery remained
`2/3` versus `1/3` passive, identical to the exact-profile control, at 720
verifier bits, 150 logical lifetimes, and 768 consumed transition rows. The
boundary is retained as a reusable memory primitive but is not promoted as a
capability gain. The next experiment must hold opaque intentions stable across
genuinely different uncertainty contexts and make transfer causally necessary;
more kernel capacity or calibration repeats alone is not justified. Evidence
is in
`session_records/factored_active_probe_context_transfer_pressure_2026-08-11/sample_efficiency_ledger.json`.

## Context-shift transfer audit (2026-08-11)

The transfer boundary was tested under an explicit distribution shift: utility
was calibrated on fresh rendered Brain Workshop lifetimes with target timing
order unshuffled, then queried on fresh lifetimes with timing order shuffled.
The candidate intentions remained opaque and the controller, decoder, factual
router, and probe request stayed read-only. This rules out a trivial
same-profile lookup explanation for any transfer signal.

At four calibration outcomes per candidate, the contextual memory still
recovered `2/3` active targets versus `1/3` passive, exactly matching the
matched exact-profile control at 720 verifier bits, 150 logical lifetimes, and
768 consumed rows. It is therefore not promoted. The evidence narrows the
problem: transferring scalar utility is insufficient when the underlying
factual probe model does not make the consequence informative enough to identify
a better intention. Evidence is in
`session_records/factored_active_probe_context_shift_pressure_2026-08-11/sample_efficiency_ledger.json`.

## Append-only trace-conditioned composition growth (2026-08-11)

The external-memory boundary now has an explicit append-only growth seam for
composition depth. `ExternalSkillFragmentGrowthCombiner` owns one stable
trace encoder and canonical readout plus zero-initialized external residual
slots. A new slot is appended for a new structural depth, trained from fresh
outcomes, and then protected. Earlier slots, the register interpreter, the
fragment bank, and the parent controller remain frozen. The slot receives the
standardized rich learner trace—learned instruction codes, transition deltas,
and opaque segment lengths—not fragment indices, operation names, verifier
metadata, or raw modality data.

This is a more faithful implementation of the CPU-plus-files idea than
allocating a separate decoder for every target: the frozen controller and
interpreter provide reusable computation, while the independently growing
external memory supplies depth-conditioned capacity. Zero initialization makes
the transaction behavior-preserving at append time, and the prefix-protection
API makes accidental updates to mastered capacity testable.

The first decisive diagnostic exposed an important acquisition constraint. A
single shared decoder trained sequentially against independently acquired
fragment representations lost the earlier atomic mappings (`rotate` 0.7474,
`complement` 0.3828, and `prefix_parity` 0.5807 after the fourth acquisition).
Jointly aligning the four atomic fragments to one shared output foundation
instead reached `[0.9219, 0.9583, 0.9922, 0.9401]` at 128 foundation updates.
That foundation is now frozen before growth. A summary-only residual slot was
also rejected; replacing it with a full trace-conditioned segment combiner
raised the 12-target depth-2 minimum from roughly `0.61` to `0.75` at 128
updates and to `0.9167` at 256 updates in the seed-69316 diagnostic. The
longer run reached perfect or near-perfect accuracy on most ordered pairs,
while retaining one shared growth combiner and one shared opaque decoder.

This is a strong bounded-growth signal, not general continual learning. The
remaining promotion pressure is deeper growth (depth 3 and 4), repeated
cross-depth retention, held-out program transfer, storage/latency scaling,
and independent-seed replication. The executable pressure test is
`experiments/external_skill_fragment_composition_amodal/train_depth_growth.py`;
its report separates training verifier bits from audit exposure and records
the frozen-parent, frozen-bank, persistence, missing-evidence, and no-replay
gates. Evidence and rejected acquisition controls belong under
`session_records/external_skill_fragment_depth_growth_2026-08-11/`.

## Cumulative protected-prefix composition growth (2026-08-11)

The first depth-growth implementation used one residual slot per exact
composition depth. That preserved mastered behavior but discarded the learned
depth-2 capacity whenever a depth-3 program ran. The growth combiner is now a
versioned external-memory ABI with cumulative protected-prefix application:
deeper traces reuse every admitted slot up to their structural depth, while a
new slot remains zero-initialized and is the only trainable capacity after the
prefix is frozen. Exact-depth application remains available as an explicit
compatibility mode for ablations.

In the matched seed-69316 rendered audit, cumulative prefix reuse preserved the
atomic/pair/trained-triple minima at `0.9896/0.9688/0.9896` with zero replay and
unchanged parent/interpreter-bank digests. Held-out triple accuracy improved
from the exact-depth baseline `[0.5729, 0.5729, 0.3229]` to
`[0.5833, 0.6042, 0.4688]`. The result is retained as a positive architectural
gain, not promoted: the held-out gate still fails, so the current bottleneck is
reusable operator algebra rather than prefix retention.

A joint depth-2 foundation curriculum was rejected in the same audit family.
Updating old shared computation while learning pairs produced strong pair and
triple fits but reduced the atomic retention minimum to `0.5208`. This is an
explicit no-replay catastrophic-forgetting failure and confirms that future
foundation expansion must be isolated behind protected external capacity or a
verified copy-on-write transaction.

## Protected serial external state diagnostic (2026-08-11)

The composition boundary now includes `ExternalSkillFragmentSerialCombiner`,
which maintains an opaque external state and applies one learned transition at
each fragment boundary. Its position-indexed and shared-transition variants
grow by zero-impact append-only slots, support protected prefixes, and persist
through a versioned checksum-verified payload. This makes the CPU-plus-files
idea an executable state-transition ABI rather than a final-readout residual.

The source-mastered seed-69316 rendered audit used the shared-transition
variant after mastering all four source primitives. Source retention remained
`0.9974` or better, but no target stable prefix was reached. Train accuracy was
`0.5286/0.8776/0.8932`; held-out accuracy was `0.6068/0.4453/0.5260`; and
wrong-order accuracy was `0.5859/0.8568/0.6953`. Missing-evidence,
reward-shuffled, frozen-parent, persistence, and zero-replay controls passed,
with `449,280` unique verifier bits, `1,472` optimizer updates, and no replay.

This is a decisive negative capability result. External serial state is now a
clean replaceable memory primitive, but final scalar outcomes still do not
assign enough credit to learn the ordered execution law. More slots, larger
state, or more decoder capacity are not justified next; the next pressure test
must expose causal prefix execution or an equivalent verifier-gated credit
path while preserving the frozen controller and no-replay accounting. The
rejected report and ledger are in
`session_records/external_skill_fragment_serial_state_rejected_2026-08-11/`.

### Direct causal-prefix verifier diagnostic (2026-08-11)

The serial ABI was extended with `forward_prefixes()`, returning one opaque
external state snapshot after every fragment boundary. A trainer-only
diagnostic then scored every snapshot against the corresponding prefix task
using fresh verifier outcomes and the shared action decoder. This was a
strictly stronger credit signal than final-only training, while keeping the
controller, register interpreter, and acquired bank frozen.

The source-mastered seed-69316 run used prefix-credit weight `0.25`. Source
retention remained at least `0.9974`, but held-out accuracy was
`0.6042/0.4271/0.5234`, no stable prefix was reached, and wrong-order
accuracy was `0.6354/0.8177/0.7240`. The run consumed `1,334,016` unique
verifier bits, including `884,736` prefix-credit bits, with zero replay. Short
matched weights `0.25`, `0.5`, and `1.0` likewise produced no stable prefix.

Direct intermediate decodability is therefore rejected as a learning
mechanism. `forward_prefixes()` remains useful infrastructure, but it does not
solve credit assignment: a prefix can be decodable without proving that its
transition caused the final ordered behavior. The next experiment must use
common-random leave-one-prefix-out interventions and train transition use from
paired scalar utility differences. Evidence is archived under
`session_records/external_skill_fragment_prefix_credit_rejected_2026-08-11/`.

### Common-random leave-one-prefix-out credit diagnostic (2026-08-11)

The next causal intervention omitted each serial transition in turn and
compared the final action outcome with the intact run under common-random
sampling. An external transition-use head gated the serial state; only paired
scalar utility differences trained that head. This is a closer match to the
actual credit question than direct prefix decodability.

The source-mastered seed-69316 full run retained all primitive files at
`0.9974` or better, but held-out accuracy was `0.6458/0.4063/0.5599`, no
stable prefix was reached, and wrong-order accuracy was
`0.5833/0.8438/0.7161`. It used `891,648` unique verifier bits, including
`442,368` leave-one-out bits, with zero replay. A matched short rung improved
mean held-out accuracy from `0.566` to `0.587`, so the mechanism remains a
useful diagnostic signal, but the full promotion gates reject it.

The causal ABI and external gate are retained. The next experiment should make
the intervention informative by selecting verifier-private sequences where
omitting a candidate transition changes the answer with high probability, then
compare that active arm against a passive paired control. More state capacity
is not justified until this signal transfers to held-out orders. Evidence is in
`session_records/external_skill_fragment_leave_one_out_rejected_2026-08-11/`.

### Active causal sequence selection diagnostic (2026-08-11)

The proposed next step was implemented as a trainer-only data-curation
mechanism. For every training update, a larger pool of fresh pixel-rerendered
candidate sequences is evaluated with common-render,
leave-one-transition-out verifier outcomes. The active arm keeps the highest
answer-changing rows per opaque target; a passive arm pays for the same probe
and keeps a matched random subset. Candidate/intervention outcomes remain
verifier-private and do not become controller inputs, combiner features, or
decoder metadata.

The three-seed matched rung used a serial combiner, leave-one-out credit weight
`0.5`, candidate multiplier `2`, updates `8/16/16`, batch size `8`, span `3`,
and audit count `16`. Active held-out order accuracy was
`0.5208/0.4792/0.5000`, `0.6042/0.4583/0.5000`, and
`0.5625/0.4375/0.6042`; passive was
`0.5208/0.4792/0.5000`, `0.6042/0.4375/0.5000`, and
`0.5625/0.4375/0.6042`. Neither arm reached a stable prefix. Each seed used
`68,688` unique verifier bits, including `41,472` selection/intervention bits,
with zero replay and `120` optimizer updates.

Active selection is therefore rejected as the current composition fix. The
selected causal signal was usually zero or unstable, so the bottleneck is not
which rows are selected; it is that the learner rarely produces informative,
answer-changing counterfactuals on held-out contexts. Retain the selection ABI
and matched accounting, but first improve verifier-gated counterfactual
sensitivity or delayed credit. Evidence is archived under
`session_records/external_skill_fragment_active_selection_rejected_2026-08-11/`.

### Stochastic multi-sample causal selection diagnostic (2026-08-11)

The active-selection probe was strengthened to use a temperature-`0.5`
stochastic policy and four common-random verifier samples per candidate. This
resolved the measurement problem: answer-changing signal became materially
larger. The active arm nevertheless failed to convert that signal into
ordered execution. Across seeds `41/42/43`, active held-out accuracy was
`0.4792/0.5208/0.5000`, `0.5833/0.4583/0.5625`, and
`0.5417/0.4792/0.4792`; the matched passive arm was
`0.5208/0.4792/0.5000`, `0.6042/0.4375/0.4792`, and
`0.5625/0.4375/0.4792`. Neither arm reached a stable prefix.

Each seed consumed `165,456` unique verifier bits, including `138,240`
stochastic selection/intervention bits, with zero replay and `120` optimizer
updates. The result rejects selection—not the probe—as the missing mechanism:
the learner can now measure answer-changing interventions, but does not learn a
reusable ordered execution law from choosing them. Retain the multi-sample
probe for future audits; the next implementation must improve the external
execution state/operator representation itself. Evidence is archived under
`session_records/external_skill_fragment_stochastic_selection_rejected_2026-08-11/`.

### Direct terminal-trace baseline (2026-08-11)

The register interpreter's terminal state was tested directly, without an
additional learned composition codec. This asks whether the learned combiner
itself is destroying ordered information. Across seeds `41/42/43`, held-out
order accuracy was `0.5208/0.4792/0.5000`, `0.5833/0.3542/0.5000`, and
`0.5208/0.4792/0.4583`. The direct trace did not produce a stable prefix or a
positive held-out transfer result. Wrong-order, missing-evidence,
reward-shuffled, and persistence controls remained structurally valid.

Each seed used `13,392` unique verifier bits and `120` optimizer updates, with
zero replay. This rejects the hypothesis that the learned composition codec
is the sole bottleneck: bypassing it does not recover ordered held-out
execution. The next implementation should therefore improve the factual
external execution/model representation rather than add another selector or
combiner variant. Evidence is archived under
`session_records/external_skill_fragment_trace_baseline_rejected_2026-08-11/`.

## Per-file fast plasticity at the executable boundary (2026-08-11)

The CPU-plus-files runtime now has an explicit path for external state that can
learn while the controller and shared interpreter are frozen. The versioned
`ExternalProgramFastCell` uses the learned controller state and the opaque
current intention as a query, reads an independently stored
`ExternalFastWeightState`, and exposes its value as a temporary `meta_context`
to the protected-meta register interpreter. The context adapter is
zero-initialized, so adding a cell is behavior-preserving until its
replaceable memory-side adapter is trained.

Every logical executable file owns its own cell state. The runtime grows and
prunes those states with the external program memory, and tensor-only runtime
checkpoints include the cell state plus the prior opaque file/query binding.
That prior binding is important: delayed scalar feedback updates the file that
produced the previous output, not whichever file happens to be selected on the
next tick. Missing evidence and failed outcomes leave the stored computation
unchanged; a successful opaque action is the only write value.

This closes the implementation seam between “memory is an isolated growing
system” and “executable files are reusable computation.” It does not yet claim
new learned computation or positive transfer: the adapter must first be
trained on a reusable capability family, then a frozen-core rendered audit
must compare a fresh cell against an inherited cell while checking complete
prefix retention, shuffled outcomes, missing evidence, route switching,
reload, and zero replay. The focused and full regression suites cover the ABI,
delayed-credit binding, mixed-file isolation, persistence, and frozen
controller contract (`721` tests passing at this checkpoint).

## Promoted external file-cell transfer (2026-08-11)

The first reusable memory-side codec audit now passes its promotion gates on
two seeds. A source codec was trained once, then frozen. Fresh target logical
files received only positive opaque action/outcome writes; the inherited codec
reconstructed the stored action at the first target lifetime on both seeds.
Matched fresh codecs required stable prefixes of 130 and 116 target lifetimes.
Source retention minima were `0.9954` and `1.0000`, and the inherited target
used zero target optimizer updates and zero primary replayed examples.

This promotes a bounded architectural capability: an isolated external memory
cell can carry reusable learned computation into newly allocated file state
without updating the controller or shared interpreter. Failed outcomes and
missing evidence are exact no-ops, delayed credit stays bound to the producing
file/query, and tensor-only persistence remains exact. The action codebook is
opaque and fixed across new logical lifetimes, so the control measures codec
transfer rather than a semantic lookup.

The boundary is intentionally not widened. This is not arbitrary procedure
invention, unrestricted growth, or general continual learning. The next
pressure test must use rendered Brain Workshop sequence lifetimes and require
complete-prefix retention, shuffled outcomes, route switching, missing
evidence, corruption, and zero-replay transfer across genuinely new rules.
Evidence is archived under
`session_records/external_program_fast_cell_transfer_2026-08-11/`.

## Causal external working-memory transfer (2026-08-11)

The working-memory boundary is now explicit in the canonical Brain Workshop
runtime as `ExternalWorkingMemoryCell`. It owns a versioned external tensor
window of learned event payloads, opaque actions, scalar outcomes, and
presence. Its causal contract is strict: it reads the current event against
the old state, emits context for action selection, and only then appends the
current row. State persistence and capacity growth preserve the newest
logical rows without touching controller weights.

The two-seed rendered audit trained the replaceable codec on fresh n-back-2
lifetime outcomes and then froze both controller and codec. Fresh external
state reached `1.0000/1.0000` on both seeds, while matched fresh controls were
`0.5000/0.5000`; shuffled-outcome and history-reset controls remained near
chance. This promotes causal memory-state transfer and corrects the earlier
post-write reconstruction measurement.

The n-back-3 probe remained near chance, so longer rule transfer is explicitly
not promoted. The next high-ROI experiment is protected external rule growth:
acquire n-back-3 in a new memory file while retaining and causally rechecking
n-back-2, then test route discovery, reversal, reload, and zero-replay
complete-prefix retention. Evidence is in
`session_records/brainworkshop_causal_working_memory_transfer_2026-08-11/`.

## Causal protected external rule growth (2026-08-11)

The follow-up now promotes the first bounded rule-growth transaction over the
new working-memory ABI. A source n-back-2 capability is trained once. A
separate `ExternalWorkingMemoryCell` is appended for n-back-3, and the source
cell, controller, and source adapters are frozen before target acquisition.
The two cells are selected by ordinary rendered cue symbols that enter through
the learned stimulus encoder; no rule ID or verifier metadata is passed to the
controller or route table. The route table receives only learned event keys,
opaque slot indices, and scalar verifier outcomes.

Across seeds `17` and `18`, all eight source retention lifetimes before and
after growth and all eight target lifetimes reached `1.0000`. Both routed
source and target rollouts reached `1.0000` accuracy and selected the intended
slot on every batch. The cue-shuffled control selected the target slot only
`0.5398` and `0.5540` of the time. Controller and source-codec digests were
unchanged; route state reloaded exactly with the compatible learned-event
encoder; patient reversal evidence changed only a copied route table; and each
seed used `64` source plus `256` target optimizer updates with zero replayed
examples. The reports contain the authoritative accounting (`73,728` verifier
bits and `10,240` logical lifetimes per seed).

This promotes a precise capability: frozen shared computation can acquire a
new causal external rule file while protecting and routing an earlier file.
It does not promote arbitrary rule induction, unrestricted memory growth,
compression, or general continual learning. A route table keyed by a learned
event representation must be restored with its compatible encoder version;
representation migration is the next ABI pressure point. Evidence and the
sample-efficiency ledger are archived in
`session_records/brainworkshop_causal_rule_growth_2026-08-11/`.

## Causal repeated depth growth and route-representation ABI (2026-08-11)

The next audit extends protected external rule growth through a three-file
prefix. It trains n-back-2, appends an isolated n-back-3
`ExternalWorkingMemoryCell`, freezes the acquired prefix, then appends an
isolated n-back-4 cell. The controller, learned event encoder, and every
earlier cell, adapter, and decoder remain frozen before each later acquisition.
Rendered cues `4`, `5`, and `6` are ordinary frontend observations; the
controller and route ledger receive only learned event tensors, opaque slot
indices, and scalar verifier outcomes.

Seeds `17` and `18` both retain all three rules at `1.0000` across eight fresh
lifetime probes and route all three cues at `1.0000`. Shuffled-cue target-slot
selection was only `0.3523/0.4489/0.2273` on seed `17` and
`0.4034/0.4631/0.2074` on seed `18`. Protected-prefix digests and the shared
controller stayed unchanged, route state reloaded exactly with its compatible
encoder, incompatible encoder state was rejected, reversal mutated only a
copied table, and all `1,152` optimizer updates used zero replayed examples.
This promotes repeated bounded rule growth, not arbitrary rule induction,
unrestricted memory growth, compression, or general continual learning. The
reports and accounting ledger are in
`session_records/brainworkshop_causal_depth_growth_2026-08-11/`.

The route-state ABI is now `brainworkshop-route-state.v2`. Because context
keys are learned event representations, the payload carries a versioned
encoder configuration and a digest of its learned state. Loading a route table
with a mismatched representation fails explicitly instead of silently falling
back to append order. This preserves the useful independence of external route
memory while making representation migration a deliberate, testable operation.

## Held-out external rule growth and outcome-only route discovery (2026-08-11)

The next causal audit trains n-back-2, n-back-3, and n-back-4 external files,
then appends an n-back-5 file under rendered cue `7`. Cue `8` is withheld from
the route ledger and introduced only after the new file is trained. The route
ledger must discover the correct opaque slot from scalar verifier outcomes;
neither the controller nor the learned event encoder may update during growth
or discovery.

Seeds `17` and `18` both passed the complete boundary. All prefix and target
retention probes reached `1.0000`; the held-out cue was absent before
discovery, became routable from outcomes, and recovered `1.0000` accuracy with
`1.0000` target-slot selection after discovery. The shared controller and
stimulus-encoder digests were unchanged, route state reloaded exactly with the
compatible encoder, incompatible encoder state was rejected, and replayed
examples remained `0`. Each run used `832` optimizer updates and recorded
`270,336` training verifier bits plus `24,864` audit bits.

This promotes held-out outcome-only route discovery over bounded external rule
growth. It does not establish arbitrary new computation, unrestricted memory
growth, compression, or general continual learning. Evidence and the
sample-efficiency ledger are archived in
`session_records/brainworkshop_heldout_rule_growth_2026-08-11/`.

## Cross-family rule growth and route hysteresis (2026-08-11)

The next pressure test varies the private verifier family rather than only
increasing n-back depth. `cross_family_rule_growth.py` acquires isolated
external files for n-back-2, pair parity, adjacent switching, and
single-symbol parity through one fixed event/intention path. The final family
is trained under cue `7`; cue `8` is withheld from the route ledger until
after training. The learner sees only rendered symbols, opaque actions, and
deterministic scalar verifier outcomes.

Seeds `17` and `18` passed complete-prefix retention, new-family mastery,
unchanged controller and event-encoder digests, held-out outcome-only route
discovery, shuffled-cue controls, exact route reload, incompatible learned
event-representation rejection, and zero replay. Held-out recovery was
`0.9978` and `0.9598` accuracy with `1.0000` target-slot selection in both
seeds. The weakest retained primitive was `0.8594`; all other retained
families reached `1.0000`.

This audit also exposed a reusable route-memory failure mode. A preferred file
was being abandoned after one stochastic scalar failure, so a competent
capability could not remain selected. The runtime now exposes external
context-route failure patience. Discovery uses patience `1` to explore the
fallback bank; exploitation uses patience `4` to require sustained evidence
before demotion. No controller, frontend, or capability weights change during
this policy update.

Each seed used `832` optimizer updates, `344,064` training verifier bits,
`59,648` audit bits, and zero replayed examples. This promotes cross-family
outcome-only route discovery over bounded external rule growth. It does not
establish arbitrary new computation, unrestricted memory growth, compression,
or general continual learning. The reports and accounting ledger are in
`session_records/brainworkshop_cross_family_rule_growth_2026-08-11/`.

## Bind-once factual execution and unknown rejection (2026-08-11)

The policy-free intention path now exposes `ExternalBoundTransitionModel`, a
memory-side view that binds one opaque contextual address before iterative
model-based execution. It preserves the factual memory's `predict_with_hit`
evidence. `ExternalModelBasedPlanner` and `PolicyFreeAmodalRuntime` accept
`require_known=True` to drop missing transition rows before beam ranking and
fail closed when no verified prefix survives. This prevents an unobserved
transition from being silently interpreted as a default state, while keeping
continuous learned models available through an explicitly unverified
compatibility path.

This is an execution-integrity and experimental-control improvement, not a
learned capability promotion. The next capability claim still requires
fresh-versus-inherited held-out curves over genuinely different transition
families, wrong/corrupted/missing-memory controls, exact retention, and equal
compute accounting.

## Outcome-gated open external-compute growth (2026-08-11)

The canonical Brain Workshop external-compute harness now allocates files
transactionally instead of preallocating the whole bank. A candidate is
trained in a fresh slot from rendered events, opaque actions, and scalar
verifier outcomes. It is admitted only after a stable direct mastery prefix;
otherwise the newest slot is rolled back and the next candidate is attempted.
Admitted files and the shared controller/frontend are frozen before later
growth, and route evidence is appended only after admission.

Across seeds `17` and `18`, five files were admitted from six candidates.
The `nback2` candidate failed stable mastery in both seeds and was rolled
back; a later `symbol_parity_odd` candidate reused that physical slot. All
five direct and routed files passed, same-context replacement reached
1.0000, old-file forced retention remained 1.0000, route reload was
exact, and replay was zero. The weakest routed-file accuracy was `0.8693`.
The corrected accounting includes the rejected candidate: each seed used
`880,128` unique training verifier bits, `23,552` audit bits, `70,272`
logical lifetimes, and `1,152` optimizer updates.

This promotes outcome-gated append-only capacity growth with failed-candidate
rollback and protected-prefix retention. It remains bounded by the candidate
schedule and does not establish unrestricted physical growth, arbitrary
program induction, learned compression, or general continual learning.
Evidence and the sample-efficiency ledger are in
`session_records/brainworkshop_external_compute_open_growth_promoted_2026-08-11/`.

## Attempted-outcome scalar credit for external n-back acquisition (2026-08-12)

The open-growth audit isolated a learning-dynamics bottleneck. The external
event window already contained enough information for n-back-2: a discarded
diagnostic probe decoded it perfectly, while the original reinforce objective
collapsed to the 75% majority-action policy. The canonical external-file
trainer now exposes an outcome-only `attempted_bce` update. It trains only
the logit for the action actually attempted against that action's deterministic
scalar verifier outcome; no correct-action or unattempted-action target is
constructed. A small entropy term preserves exploration.

Across seeds `17` and `18`, five files were admitted and routed, including
n-back-2 at 1.0000 direct accuracy in both seeds. The weakest routed-file
accuracy was 0.8828 and 1.0000; same-context reversal and old-file
retention were 1.0000, route reload was exact, and replay was zero. A
matched shuffled-feedback n-back-2 control remained below mastery at maximum
0.4479 and 0.2760, supporting causal use of attempted scalar outcomes.

This promotes a reusable scalar-credit mechanism for one external
working-memory capability. It does not establish arbitrary program induction,
unrestricted growth, consolidation, or general continual learning. Evidence
and separate control accounting are in
`session_records/brainworkshop_external_compute_nback2_credit_promoted_2026-08-12/`.

## Deeper n-back growth and event-window capacity (2026-08-12)

The next pressure test kept the controller, learned event frontend, opaque
external-file interface, attempted-outcome credit, route reversal, and
shuffled-feedback control fixed while adding only a generic event-window
parameter. The private verifier now constructs `nbackN` targets from the
requested depth; no n-back-specific reasoning branch is exposed to the
learner.

With four event tokens, n-back-3 reached stable `1.0000` direct accuracy on
both tested seeds. A matched n-back-4 capacity probe remained below the `0.80`
mastery threshold on every fresh lifetime (`0.7781/0.7531/0.7594/0.7250` and
`0.7531/0.7594/0.7250/0.7906`). With the same basis and a five-event window,
n-back-4 reached `1.0000` on every fresh lifetime on seeds `17` and `18`.

The full five-event open-growth runs admitted and routed eight files, including
n-back-2, n-back-3, and n-back-4. Direct stable mastery, route selection,
same-context replacement, old-file retention, exact route reload,
shuffled-feedback rejection, frozen controller/frontend, unchanged admitted
files, and zero replay all passed. This is evidence for an information-window
boundary and a domain-general representation extension, not a claim that
memory is already unbounded.

The new boundary remains explicit: the external history is still a bounded
window, and route identity still requires distinct learned context keys. The
next architecture task is a scalable external history/memory contract that
can compress or retrieve beyond the window while preserving causal credit,
binding, and no-replay retention under capacity pressure. Full reports and
separate accounting are archived in
`session_records/brainworkshop_external_compute_deeper_nback_promoted_2026-08-12/`.

## External temporal-history memory contract (2026-08-12)

`ExternalTemporalHistoryMemory` is the first storage-side step beyond a
rolling event window. It owns scoped append-only learned-event records,
monotonic sequence positions, explicit missing-history masks, checksummed
payloads, and isolated scope clearing. Its only retrieval primitive is an
opaque relative-offset query; the controller never receives physical record
IDs or task metadata.

The ABI probe stored 128 records per scope, read offsets through 127 exactly,
reloaded with zero value difference, preserved the other scope after a clear,
and rejected a corrupted payload. Because the probe used no verifier bits,
optimizer updates, or learned address selector, this qualifies storage and
state isolation only. It does not qualify useful learned addressing or
general continual learning.

The next high-value experiment is an external offset/address selector trained
from scalar outcomes. It must show that the selector discovers useful history
addresses, survives wrong-offset and missing-history controls, and improves a
held-out learning curve without replaying mastered sequences. Evidence is in
`session_records/brainworkshop_external_temporal_memory_contract_2026-08-12/`.

## Outcome-only learned temporal offset growth (2026-08-12)

The first learned user of the temporal store is an external offset policy. It
selects one positive relative offset from an opaque categorical distribution,
reads that learned event tensor, and receives scalar policy credit alongside
attempted-output credit. The controller and event frontend remain frozen.

Across seeds `17` and `18`, a fresh n-back-5 file selected offset `5` on all
eight fresh lifetimes and reached minimum accuracies `1.0000` and `0.9132`.
The previously mastered n-back-4 file remained at `1.0000`; forced wrong-offset
accuracy stayed below `0.685`, missing-history stayed below the `0.80` mastery
threshold, shuffled-outcome controls stayed below `0.660`, and replay was zero.

This qualifies scalar-credit discovery of one global relative offset in an
isolated external file. It does not qualify conditional addressing, content
search, multiple competing memories, learned compression, or general
continual learning. The next pressure test must make the address policy
context-conditioned and force it to choose among multiple useful offsets or
content keys while protecting earlier files.

Evidence and accounting are archived in
`session_records/brainworkshop_external_temporal_offset_growth_promoted_2026-08-12/`.

## Context-conditioned temporal route growth

The next composition uses the existing memory-side
`PersistentOpaqueContextRouteEvidence` table to select among isolated
external temporal capability files. The route query is a normalized learned
event tensor, not a task identifier. Each selected file reads its own
append-only temporal history and learns its own relative offset from scalar
episode outcomes. The controller and event encoder remain frozen; the old
file is protected before the new file is acquired and no old sequence is
replayed.

The two-seed promotion at cues `11` and `12` learned n-back-4 and n-back-5
files with offsets `4` and `5`. Both contexts selected the correct file on
every evaluated lifetime; routed target accuracies were `1.0000` and
`0.9514`, while the retained source stayed at `1.0000`. Unknown-context
fallback, wrong-file, wrong-offset, missing-history, shuffled route feedback,
exact route reload, frozen-core, and zero-replay controls passed.

This is bounded context-conditioned external routing, not general continual
learning. The unresolved pressure point is same-context binding of multiple
useful addresses (or content keys), followed by learned compression and
capacity pressure. Evidence and accounting are archived in
`session_records/brainworkshop_external_temporal_context_route_growth_promoted_2026-08-12/`.

## Same-cue query-conditioned temporal address growth

The stronger address test holds the rendered cue constant and varies only a
learned query event. A source external capability file first acquires a
generic temporal readout and offset 4 from scalar outcomes. The file is then
frozen. A memory-side context-keyed address table must acquire offset 5 for a
new query event, while retaining offset 4 for the old query, without replay or
readout updates.

Across seeds `17` and `18`, both query-conditioned addresses selected their
correct offsets on every retained lifetime and both source and target reached
`1.0000`. Unknown-query fallback, wrong-offset, missing-history,
shuffled-outcome, exact reload, frozen readout/controller/frontend, and zero
replay all passed. This qualifies bounded same-cue multi-address acquisition
through learned event keys. It does not yet qualify content search over
unseen-but-related keys, learned compression, unrestricted memory growth, or
general continual learning. Evidence and accounting are archived in
`session_records/brainworkshop_external_temporal_query_address_growth_promoted_2026-08-12/`.

## Related-key temporal content retrieval (2026-08-12)

The next composition uses the canonical persistent
`AppendOnlyContentAddressedMemory` rather than a route table alone. Two
learned event keys are stored with opaque external capability-address values;
the source capability file is frozen before the target route is acquired, and
the controller and event encoder remain frozen throughout retrieval. Exact
queries and nearby learned-event queries, produced by a fixed 20% normalized
perturbation, must recover offsets 4 and 5 through cosine content addressing.

Across seeds `17` and `18`, source and target reached `1.0000` for both exact
and related-key reads. The memory returned an explicit no-hit for an unknown
key, preserved related-key routes across an exact reload, removed hits after
clear, and rejected a checksum-corrupted snapshot. The controller, event
encoder, and capability file digests were unchanged and replay was zero. This
is a useful ABI and retrieval result, but it remains bounded content-addressed
composition:
it does not qualify learned compression, capacity management, arbitrary new
computation, or general continual learning. Evidence and accounting are
archived in
`session_records/brainworkshop_external_temporal_content_retrieval_growth_promoted_2026-08-12/`.

## Verified external temporal-memory compaction (2026-08-12)

The append-only memory now exposes a versioned, scope-safe
`replace_from_candidates` commit boundary. A memory-side policy can propose a
rewrite, while an independent verifier evaluates the candidate before the
memory adopts it. The expected store version prevents a stale verifier from
overwriting newer state, and persistent stores snapshot the accepted rewrite
atomically.

The two-seed pressure test writes three learned records: an exact source key,
a nearby source alias with the same opaque capability address, and an exact
target key. A held-out route verifier accepts the source/alias merge and
rejects a source/target merge before mutation. Seeds `17` and `18` both kept
exact and related-key source and target accuracy at `1.0000` after saving one
row. Reload, checksum corruption, stale-version, frozen-controller,
frozen-encoder, frozen-capability-file, and zero-replay gates passed.

This qualifies safe compaction of redundant learned content keys, not arbitrary
compression. Distinct capabilities still need distinct representational
capacity; learned pair selection, multi-row compression, capacity scheduling,
and general continual learning remain open. Evidence and accounting are
archived in
`session_records/brainworkshop_external_temporal_verified_compaction_growth_promoted_2026-08-12/`.

## Learned live-memory compaction selection (2026-08-12)

The existing `OpaqueConsolidationPolicy` now composes with the canonical
persistent append-only memory rather than ending at an isolated synthetic bank
audit. It is trained from scalar duplicate-rewrite utility, then receives only
learned event keys, learned values, strength, and relative age for a live
three-row memory. A source key and nearby source alias share one opaque
capability address; a target key carries a different address. The learned
policy must select the redundant pair under every physical row permutation
before the independent route verifier permits a rewrite.

Across seeds `17` and `18`, the learned policy selected the redundant pair on
all six permutations, while the untrained policy selected it twice. Both live
compactions saved one row, passed reload and checksum-corruption controls, and
kept the controller and event encoder frozen with zero replay. This closes the
manual-proposal gap for this narrow redundancy case. It does not establish
end-to-end capability acquisition or arbitrary compression: distinct skills
still require distinct representational capacity, and learned multi-row
compression, capacity scheduling, and general continual learning remain open.
Evidence and accounting are archived in
`session_records/brainworkshop_external_temporal_learned_compaction_growth_promoted_2026-08-12/`.

## Learned temporal capacity scheduling (2026-08-12)

The canonical persistent temporal-memory path now composes with the generic
`OpaqueCapacityPlanner`. `MemoryCandidates.pad_to_capacity()` provides a
typed fixed-budget policy view for variable-capacity stores: padding is
unoccupied and zero-filled, while the append-only backend and controller
remain unchanged. Planner proposals remain advisory; only the independent
route verifier can authorize `replace_from_candidates`, followed by a
version-checked atomic persistence transaction.

The two-seed pressure test trains the planner from scalar utility on generic
candidate banks, then transfers it to a live four-row memory. Two distinct
learned event-key addresses each arrive with a redundant alias. Two new
addresses must therefore be admitted only after a learned consolidation
selection. The same stream is run after a physical row-order reversal.

Across seeds `17` and `18`, held-out utility for admit, evict, consolidate,
and grow was `1.0`; fresh consolidation controls reached only `0.15625` and
`0.25`. Both forward and reversed streams completed two verifier-approved
compactions and two admissions, retained four distinct routes after every
stage, reloaded exactly, rejected checksum corruption, kept the controller
and event encoder byte-stable, and used zero replay. The full reports and
ledger are archived in
`session_records/brainworkshop_external_temporal_capacity_schedule_promoted_2026-08-12/`.

This promotes bounded replay-free capacity scheduling and sequential
verifier-gated multi-row compaction in the canonical temporal-memory path. It
does not establish arbitrary shared-structure compression, semantic
equivalence discovery, unbounded memory, autonomous verifier design, or
general continual learning. The next pressure test is a learned
multi-row/shared-structure representation that reduces physical storage for
genuinely distinct but compositional capabilities, followed by longer
nonstationary streams and retention-adjusted transaction regret.

## Shared-basis external value compression (2026-08-12)

`SharedBasisContentAddressedMemory` adds a replaceable factorized-value
backend to the canonical memory boundary. Keys, logical rows, strengths,
timestamps, and occupancy remain independent; only the value payload is
represented as per-row coefficients over an external orthonormal basis. Reads
materialize the ordinary `MemoryRead` contract, so the controller does not
depend on the storage representation.

Compression is copy-on-write and verifier-gated. A candidate rank reduction
must preserve every protected route and value, and an expected store version
prevents a stale verifier from overwriting newer state. Persistent accepted
replacements are checksummed and atomically reloadable. The backend grows its
basis online, but never silently reduces it during writes.

The two-seed canonical pressure test stores twelve distinct opaque learned
values with a shared two-dimensional structure and small residuals. A rank-one
candidate is rejected; a rank-two candidate is accepted in both forward and
reversed physical row order. Physical basis/coefficient storage falls from
`336` to `56` scalars while logical record count remains `12`; all routes,
reload, corruption, stale-version, frozen-core, and zero-replay gates pass.

This promotes safe shared-structure storage compression, not learned rank
selection, semantic equivalence discovery, arbitrary new computation,
unrestricted memory growth, or general continual learning. The rank choice is
deterministic SVD in this audit. The next pressure is an outcome-trained
structure/rank proposal on evolving residuals and long nonstationary streams.
Evidence and accounting are archived in
`session_records/brainworkshop_external_temporal_shared_basis_compression_promoted_2026-08-12/`.

## Outcome-trained shared-basis policy growth (2026-08-12)

`OpaqueSharedBasisCompressionPolicy` is a replaceable external selector that
scores a runtime-sized set of candidate representations from generic
rank/error/storage statistics. It receives one scalar verifier utility per
proposal and emits only a candidate index; it does not receive task labels,
semantic targets, or raw modality data. The selected candidate remains subject
to the memory backend's independent route/value verifier and versioned
copy-on-write commit.

The two-seed canonical audit trains on fresh generic low-rank banks, then
transfers the policy to a frozen event boundary with six old rank-two values.
It selects rank `2`, protects those routes, admits six new rank-four successor
values, and selects rank `4` without replaying the old cohort. Held-out
rank-selection accuracy is `0.875/1.000/1.000` for ranks `1/2/4` on both seeds;
fresh policies are weaker on rank one. Forward and reversed physical order,
old/new retention, reload, stale-version, corruption, frozen-core, and
zero-replay gates pass.

This promotes a narrow outcome-trained external compression preference and one
nonstationary growth transfer. It does not establish online semantic structure
discovery, unrestricted memory growth, arbitrary computation, or general
continual learning. The policy still receives a precomputed reconstruction
error feature, and only one successor transition is tested. The next pressure
is repeated online structure discovery and reversal without that precomputed
candidate-quality shortcut.
Evidence and accounting are archived in
`session_records/brainworkshop_external_temporal_shared_basis_policy_growth_promoted_2026-08-12/`.

## Raw-value shared-structure policy growth (2026-08-12)

`OpaqueSharedBasisStructurePolicy` removes the previous selector's
precomputed candidate-reconstruction-error feature. It receives only opaque
value rows, an occupancy mask, and runtime candidate ranks. Its fixed-width
singular-spectrum summary is permutation-invariant over physical rows; the
policy learns a candidate preference from one scalar verifier utility at a
time and emits only a candidate index. The memory-side verifier still owns
route/value retention, expected-version checks, and persistent copy-on-write
commit.

Across seeds `17` and `18`, held-out rank-selection accuracy was
`0.9375/1.0000/1.0000` and `0.9219/1.0000/1.0000` for ranks `1/2/4`. Fresh
controls were weaker, especially on rank 1. In the frozen canonical stream,
the policy selected rank `2` for six old values, then rank `4` after six
rank-four successor values arrived. Old and new routes survived in forward and
reversed physical order; reload, stale-version, checksum-corruption,
frozen-core, and zero-replay gates passed.

The promotion required `50,000` unique scalar-utility updates per seed. A
10,000/20,000-update calibration was retained as rejected evidence because
seed 17 did not clear the rank-one `0.80` floor. This makes scalar-feedback
sample efficiency the next bottleneck. The result is a narrow external
structure-selection transfer, not semantic structure discovery, arbitrary new
computation, unrestricted memory growth, repeated growth/reversal, or general
continual learning. Evidence and accounting are archived in
`session_records/brainworkshop_external_temporal_shared_basis_structure_growth_promoted_2026-08-12/`.

## Repeated raw-value shared-structure growth (2026-08-12)

The v2 `OpaqueSharedBasisStructurePolicy` adds a bounded
row-permutation-invariant pairwise summary to the singular spectrum it already
computes. It still consumes only opaque value rows, occupancy, and runtime
candidate ranks; unoccupied padding is ignored, and precomputed candidate
reconstruction error remains outside the policy ABI. The policy receives one
scalar verifier utility per fresh bank and emits only a candidate index.

The stronger two-seed canonical stream presents four cohorts with structure
`rank 2 → rank 4 → rank 4 → rank 4`. At every transition, the memory backend
performs an independent route/value retention check, expected-version check,
and copy-on-write persistence. Both forward and reversed physical insertion
orders are tested.

At `20,000` unique scalar-utility updates per seed, held-out rank-1/2/4
accuracy was `0.8594/0.9844/1.0000` and `0.9219/1.0000/1.0000`; both fresh
controls were weaker. Both seeds selected `2 → 4 → 4 → 4`, retained all 24
routes after each transition, and passed reload, stale-version,
checksum-corruption, frozen-core, and zero-replay gates. A 3,000-update
calibration passed live safety but failed held-out transfer and remains
rejected in the archive.

This promotes repeated bounded external structure-policy transfer, not
semantic structure discovery, arbitrary new computation, unrestricted memory
growth, regime reversal, or general continual learning. The next pressure is
genuinely changing or competing subspaces, reversal controls, and reducing
scalar-feedback cost below 20,000 unique banks. Evidence and accounting are
archived in
`session_records/brainworkshop_external_temporal_shared_basis_repeated_growth_promoted_2026-08-12/`.

## Competing-subspace dynamic-rank growth (2026-08-12)

The next pressure test expands the runtime candidate set to `(2, 4, 8)` and
feeds four rank-two cohorts from distinct orthogonal subspaces into the frozen
external memory. Their union changes from rank 2 to rank 4 to rank 6 to rank 8,
requiring `2 → 4 → 8 → 8`. The v2 policy still receives only opaque value rows,
occupancy, and candidate ranks; reconstruction error remains verifier-private.

Across seeds `17` and `18`, held-out rank-2/4/8 accuracy was
`0.9688/1.0000/1.0000`. Both seeds selected the expected dynamic sequence in
all four combinations of subspace-arrival order and physical row order. All
16 commits per seed were accepted, all 24 routes survived every prefix,
storage ended at `512/768` physical/dense value scalars, and reload,
stale-version, corruption, frozen-core, and zero-replay gates passed.

This promotes bounded dynamic-rank selection under competing subspaces. It
does not establish removal/replacement, semantic structure discovery,
arbitrary new computation, unrestricted memory growth, or general continual
learning. The next pressure is true regime reversal with replacement and
capacity pressure. Evidence and accounting are archived in
`session_records/brainworkshop_external_temporal_shared_basis_competing_subspaces_promoted_2026-08-12/`.

## Verifier-gated shared-basis regime replacement (2026-08-12)

The shared-basis backend now exposes a separate `shared_basis_rewrite_v1`
boundary for changing the logical row set of one external scope. A rewrite
candidate may remove stale rows and admit new rows, but it cannot mutate the
live memory until an independent retention probe and expected-version check
pass. Other scopes are copied unchanged, and persistent accepted rewrites are
atomically checksummed.

The two-seed canonical audit protects six source routes and gives a working
scope twelve routes from two incompatible rank-two subspaces. The policy first
selects rank `8`; the working scope is then replaced by twelve new routes in a
different rank-two regime, and the policy selects rank `4`. Both seeds retained
all protected routes, removed all old working addresses, admitted all new
working addresses, reduced storage from `272` to `136` value scalars, and
passed forward/reversed, reload, stale-version, corruption, frozen-core, and
zero-replay controls.

This promotes bounded verifier-gated regime replacement and capacity reuse. It
does not establish learned change-point detection, semantic regime discovery,
arbitrary new computation, unrestricted memory growth, or general continual
learning. Evidence and accounting are archived in
`session_records/brainworkshop_external_temporal_shared_basis_regime_replacement_promoted_2026-08-12/`.

## Learned external regime trigger (2026-08-12)

`OpaqueRegimeChangePolicy` is a separate external state policy for deciding
whether an opaque current working bank should be kept or replaced by an
incoming bank. It computes permutation-invariant spectral and cross-bank
structure features, learns from one scalar verifier utility per pair, and
emits only `keep` or `replace`. Regime IDs, task labels, semantic metadata,
and candidate reconstruction error are outside its interface.

The two-seed canonical audit first presents stable evidence and requires an
exact no-op: neither memory bytes nor the persistent store version may change.
A structurally shifted bank then triggers a verifier-gated scope rewrite while
protected routes remain intact. Both seeds reached `1.0000` held-out stable
keep and shifted replace accuracy after 1,000 detector updates; fresh controls
averaged `0.5000` because each missed a different class. Reload, stale-version,
checksum-corruption, frozen-core, and zero-replay controls passed.

This promotes a narrow learned trigger and safe replacement boundary, not
autonomous semantic change-point discovery, unrestricted memory growth,
arbitrary new computation, or general continual learning. Evidence and
accounting are archived in
`session_records/brainworkshop_external_temporal_shared_basis_learned_regime_trigger_promoted_2026-08-12/`.

## Alternating hidden regimes with protected scopes (2026-08-12)

The next pressure test runs the learned external trigger through five hidden
working-regime reversals, `A → B → A → B → A → B`, while three protected
scopes remain in the same persistent memory. The detector receives only the
current and incoming opaque value banks; before each boundary, a stable copy
must be an exact keep/no-op and the shifted bank must trigger replacement.
Every replacement uses fresh opaque addresses, so prior working routes must
be removed rather than silently shadowed.

Across seeds `17` and `18`, all five boundaries were detected and accepted in
both forward and reversed physical row order. The logical store stayed at 26
records and factorized physical value storage stayed at 168 scalars versus
416 dense scalars at all six checkpoints. Protected routes, reload, stale
version, corruption, frozen-core, and zero-replay controls passed.

This promotes bounded repeated reversal and capacity reuse, not autonomous
semantic change-point discovery, unrestricted memory growth, arbitrary new
computation, or general continual learning. Evidence and accounting are
archived in
`session_records/brainworkshop_external_temporal_shared_basis_alternating_regimes_promoted_2026-08-12/`.

## Gated residual online adaptation (2026-08-12)

The single-policy online adaptation calibration exposed catastrophic
forgetting: partial-overlap replacement improved, but stable keep collapsed.
The promoted boundary therefore freezes the existing external detector and
adds a zero-initialized residual policy. Online scalar utilities update only
the residual. Deterministic inference falls back to the frozen detector unless
the residual has positive, stronger evidence for its preferred action.

Across seeds `17` and `18`, 144 fresh online utilities raised held-out
partial-overlap replacement from `0.0156` to `0.8203/0.8906`, while stable keep
and disjoint replacement remained `1.0000`. Exact stable and fully shifted
retention, frozen controller/encoder, and zero-replay controls passed. The
negative single-policy result is archived with the promotion evidence.

This promotes parameter-isolated residual growth as one anti-forgetting
mechanism. It does not establish arbitrary residual-slot routing, unrestricted
external growth, arbitrary new computation, or general continual learning.
Evidence and accounting are archived in
`session_records/brainworkshop_external_temporal_regime_policy_online_adaptation_promoted_2026-08-12/`.

## Opaque binding-routed residual slots (2026-08-12)

The geometry-only regime policy is intentionally invariant to joint orthogonal
rotations, but that also means it cannot route two distinct capabilities whose
current/incoming relational geometry collides. `GatedResidualRegimePolicyBank`
adds a separate opaque binding-context contract. External state supplies a
learned context key; cosine routing selects an independent residual slot, and
the frozen detector remains the fallback. The bank does not interpret the key
or receive a task label.

Across seeds `17` and `18`, slot A was learned first and slot B second from 72
fresh scalar utilities each. Slot B was unchanged during slot-A learning and
slot A was unchanged during slot-B learning. After both phases, partial-shift
replacement was `0.9688/0.8828` for A and `0.9375/0.8516` for B; stable keep
and disjoint replacement remained at least `0.9766` and `1.0000` respectively.
The controller, event encoder, and base detector remained frozen with zero
replay.

Each slot was frozen only after its independent retention probe passed. The
bank rejected later updates to frozen slots and rejected a third allocation at
the configured two-slot capacity. The 96-update overadaptation rejection is
retained as evidence that promotion/stopping must be part of the lifecycle.

The full-bank lifecycle now also supports copy-on-write slot replacement. An
unsafe candidate is rejected without mutating live state; an independently
verified candidate reuses one full slot for a new opaque binding while another
binding remains retained. This is the first safe capacity-reuse boundary for
residual slots, not learned redundancy discovery or general continual
learning.

This promotes isolated opaque binding routing, not autonomous binding
discovery, unrestricted slot growth, arbitrary skill composition, or general
continual learning. The 96-update overadaptation rejection is retained as
evidence that slot allocation needs verifier-gated stopping and consolidation.
Evidence and accounting are archived in
`session_records/brainworkshop_external_temporal_regime_policy_binding_slots_promoted_2026-08-12/`.

## Learned full-bank maintenance choice (2026-08-12)

The residual lifecycle no longer requires the victim slot to be selected by
the experiment. `ExternalCapabilityEvictionPolicy` ranks variable opaque
candidate slots from generic binding-key, reliability, and age telemetry and
learns from one scalar verifier utility per fresh bank. Candidate order is
permuted independently; semantic names, task labels, physical indices, and
verifier targets remain outside the policy input.

Across seeds `17` and `18`, held-out selection reached `0.9648/0.9258` versus
fresh controls `0.3477/0.3516`. Forward and reversed candidate orders selected
the weak slot, verifier-gated copy-on-write reuse retained the sibling, and the
new binding learned after reuse. The controller and event encoder remained
frozen with zero replay.

This promotes bounded learned maintenance choice, not autonomous redundancy
discovery, universal eviction economics, unrestricted memory growth, or
general continual learning. Evidence and accounting are archived in
`session_records/brainworkshop_external_temporal_regime_policy_learned_maintenance_promoted_2026-08-12/`.

## Routed nonstationary maintenance residuals (2026-08-12)

The stationary learned maintenance selector is insufficient when the verifier
changes what “disposable” means: reliability-trained selection reached only
chance-level accuracy on an age-dominated objective. The external boundary now
supports `GatedResidualCapabilityEvictionPolicyBank`, which freezes the base
scorer and routes independently trained residual scorers by opaque context
keys. Unknown contexts fall back to the base; promoted residuals can be
activated and frozen independently.

Across seeds `17` and `18`, reliability and age residuals both transferred at
least `0.8438` in forward and reversed candidate order. Unknown contexts
retained base reliability selection at `0.9570/0.9844`, and the first residual
remained unchanged while the second learned. Controller/encoder and replay
gates passed.

This promotes bounded nonstationary maintenance adaptation, not autonomous
context-key discovery, universal maintenance economics, unrestricted growth,
or general continual learning. Evidence and accounting are archived in
`session_records/brainworkshop_external_temporal_regime_policy_nonstationary_maintenance_promoted_2026-08-12/`.

## Learned episodic binding discovery (2026-08-12)

The next bottleneck was that residual maintenance slots still received their
opaque context keys from outside. `EpisodicBindingRouter` now provides the
missing external boundary: it encodes learned event/action/outcome episodes,
provisions opaque slot keys from observed contexts, and adapts the context
encoder from only the scalar utility of the slot that was actually attempted.
No task label, correct unattempted slot, protocol field, or controller update
enters this path.

The two-seed promoted audit used 1,000 fresh route utilities per seed with zero
replay and zero controller updates. Forward route accuracy was `1.0000` on
both seeds; independent candidate-order permutation was also `1.0000` on both.
Freezing the router and exact state reload preserved the result, while
reward-shuffled controls stayed at balanced `0.5000`. This promotes a bounded
learned binding-discovery primitive, not autonomous ontology formation,
unrestricted external growth, or general continual learning. The next test is
online multi-slot discovery plus verifier-gated replacement under capacity
pressure.

Evidence is archived at
`session_records/brainworkshop_external_temporal_learned_binding_routing_promoted_2026-08-12/`.

## Online episodic binding capacity and safe replacement (2026-08-12)

The first learned binding audit exposed a real limitation: argmax routing can
remain correct while its cosine score is too poorly calibrated to distinguish
an unseen binding, and a frozen encoder can map a novel binding onto a retired
one. `EpisodicBindingRouter` v3 addresses this with two explicitly separated
paths. A scalar-trained episodic route embedding selects among learned opaque
keys; an immutable generic episode signature supplies novelty evidence and is
stored alongside each key. The signature is built only from learned event
content, opaque actions, scalar outcomes, and presence, not semantic labels or
protocol fields.

`slot_replacement_candidate()` and `replace_slot_from_candidate()` now support
key consolidation and bounded capacity reuse as copy-on-write transactions.
The external verifier must retain every protected sibling and master the new
binding before a replacement commits; rejected candidates leave the live bank
byte-stable. The controller, learned event encoder, and promoted route
encoder remain frozen during online growth.

The two-seed promoted audit used 1,000 fresh route updates per seed. Both seeds
reached `1.0000` initial and consolidated known rates, `0.0000` known rate for
the novel binding before admission, `1.0000` retained-sibling and new-binding
accuracy after replacement, `0.0000` known rate for the retired binding, and
`1.0000` permutation/reload accuracy. The learned-route reward-shuffled null
was `0.5000`. Accounting records `6,608` unique verifier bits per seed, `512`
explicitly replayed reload diagnostics, and zero replayed training examples.

This promotes bounded online binding discovery and retention-safe capacity
reuse, not unrestricted growth, autonomous ontology formation, arbitrary new
computation, or general continual learning. Evidence and checksums are
archived at
`session_records/brainworkshop_external_temporal_online_binding_capacity_promoted_2026-08-12/`.

## Learned victim choice for episodic binding capacity (2026-08-12)

The online capacity boundary no longer requires the experiment to nominate the
victim slot. `ExternalCapabilityEvictionPolicy` receives an incoming opaque
episode signature and generic candidate telemetry only; it learns a
disposability score from the scalar utility of the slot actually attempted.
Physical slot indices, semantic names, correct unattempted rows, and task
labels remain outside the policy. A separate verifier protects sibling A and
commits a replacement only when the new binding C and every protected route
pass held-out probes.

Across two seeds, held-out victim transfer was `0.8047/0.8184`, while
reward-shuffled controls were `0.3086/0.3301`. Forward and reversed candidate
orders both selected the weak slot, and both copy-on-write transactions
retained the sibling and acquired the new binding. The controller and learned
event encoder were unchanged. This promotes bounded learned maintenance
choice, not universal eviction economics or general continual learning.

Evidence and accounting are archived at
`session_records/brainworkshop_external_temporal_learned_binding_victim_selection_promoted_2026-08-12/`.

## Growable external episodic binding archive (2026-08-12)

The active episodic router must not also be the long-term archive. The new
`EpisodicBindingArchive` stores immutable learned context/signature records and
generic scalar reliability and recency evidence outside the frozen controller.
Active-slot replacement changes residency only; it does not erase the old
record. Signature lookup can therefore find a returned capability and
verifier-gated copy-on-write can reactivate it without replaying its previous
training stream.

The repeated-interleaving audit introduced four anonymous bindings through six
replacement cycles across seeds 17 and 18. It passed protected-resident
retention, zero avoidable evictions, twelve active no-op probes, archive/reload
integrity, frozen-core, and zero-replay gates. This is a promoted bounded
external-memory lifecycle result. It does not establish unrestricted archive
growth, arbitrary new computation, compression quality, or general continual
learning. Evidence and the accounting ledger are archived at
`session_records/brainworkshop_external_temporal_interleaved_binding_archive_promoted_2026-08-12/`.

The archive contract is now hardened for larger external memory. Version 2
caches normalized signature rows and exposes batched lookup, records explicit
protection/reversal state, and rejects modified serialized payloads with a
canonical SHA-256 checksum. The two-seed 1,024-record audit passed known and
unknown retrieval, query-order invariance, reload, protected-sibling reversal,
corruption, frozen-core, and zero-replay gates. This is a promoted storage and
integrity result; it does not establish learned acquisition of 1,024
capabilities, arbitrary compression, or general continual learning. A compact
tensor snapshot reduced the 1,024-record artifact from about 645 KB JSON to
about 166 KB while preserving retrieval and checksum rejection; this is
representation compaction, not semantic learned compression. Evidence and
accounting are archived at
`session_records/brainworkshop_external_temporal_archive_scale_reversal_promoted_2026-08-12/`.

## Episodic executable-artifact reactivation (2026-08-12)

The archive-to-capability seam is now explicit. `EpisodicBindingArtifactIndex`
stores one opaque external artifact handle per immutable episodic
context/signature record; it does not own executable tensors or interpret
them. `reactivate_verified()` stages a candidate active residency through
copy-on-write, refuses to displace a protected resident, and commits only
when a caller-owned held-out retention probe passes without mutating the
candidate. The external program memory remains independently versioned and
replaceable.

`episodic_artifact_reactivation.py` exercised this seam with four cold
executable artifacts and a two-slot hot cache across seeds 24101 and 24102.
Both seeds reactivated two cold artifacts, revisited an old artifact after
multiple swaps without replay, preserved a protected artifact, rejected
failed/mutating/missing/corrupt candidates without write, passed index and
executable-memory reload, and kept the shared interpreter byte-identical.
Online optimizer updates and replayed examples were zero; the audit charged
512 held-out verifier bits per seed.

This promotes bounded replay-free reactivation of external capability files,
not unrestricted memory growth, automatic synthesis of new computation, or
general continual learning. Evidence and accounting are archived at
`session_records/brainworkshop_episodic_artifact_reactivation_promoted_2026-08-12/`.

## Bounded external recipe sequence compilation (2026-08-12)

The corrected recipe boundary now distinguishes a missing atomic primitive from
a sequence-search problem. `RecipeBasis.sequence_probe()` searches bounded
opaque instruction sequences by their finite register effects and merges
equivalent prefixes. With per-slot arithmetic domains, it finds the
two-valued toggle as two ordinary increments, `INC(0, m=2); INC(1, m=2)`; the
old global-modulus implementation remains a structural mismatch and is
rejected by the domain contract.

The compiler is fail-closed: a complete search through the configured bound
returns `inexpressible`, while an interrupted search returns
`budget_exhausted`. This is a reusable external-execution foundation, not a
promotion of arbitrary learned program induction. The next decisive audit is
to compare bounded sequence compilation with stochastic outcome-only proposal
search at matched verifier cost, then store a discovered sequence as an
external file and test frozen-core retention, held-out transfer, persistence,
and no-replay acquisition.

## Outcome-only external recipe-file bridge (2026-08-12)

The generic recipe basis now has a replaceable file boundary. An external
sequence search proposes opaque edits and receives only scalar verifier
outcomes; a separate memory bank admits stable candidates transactionally,
protects earlier files, and reloads them through a checksummed payload. Search
statistics are aggregate-only. Candidate history is scoped by an opaque
external binding key, preventing a rejected candidate in one context from
poisoning a later context while allowing generic edit priors to transfer.

The two-seed audit retained a source plus two acquired files, learned an
order-sensitive target, rejected reversed execution and shuffled feedback, and
passed persistence and zero-replay controls. It did not promote a learning
curve gain from the shared scalar edit prior: the warm target search was slower
than the fresh control on both seeds. This is a useful boundary result, not a
general continual-learning claim. The next mechanism must learn
context-conditioned instruction/position proposal credit with an exploration
floor before larger program frontiers are justified.

## Context-conditioned outcome-only proposal credit (2026-08-12)

The external recipe boundary now includes a replaceable
`OpaqueContextRecipeProposalMemory`. It receives only an opaque context key,
an opaque content-addressed candidate digest, and a scalar verifier quality.
It stores aggregate credit outside the controller, keeps candidate-history
scope separate from context, persists with a checksum, and enforces a nonzero
exploration floor so inherited evidence cannot make a novel candidate
unreachable.

The two-seed audit trained two contradictory order-sensitive recipe targets in
separate opaque contexts, reloaded the policy, and reacquired both targets in
new logical lifetimes with zero replay. Held-out accuracy stayed at `1.0000`.
Warm/fresh proposal ratios were `0.1111` and `0.1176` on seed `17`, and
`0.1765` and `0.1818` on seed `18`. Each context preferred its own candidate,
an unseen context remained unbiased with the floor active, shuffled feedback
was rejected, and controller optimizer updates remained zero. Evidence and
the sample-efficiency ledger are archived in
`session_records/recipe_context_conditioned_proposal_credit_promoted_2026-08-12/`.

This promotes only bounded replay-free contextual candidate reuse. It does not
yet provide factorized instruction/position credit, related-context transfer,
automatic program synthesis beyond the bounded neighborhood, unrestricted
memory growth, or general continual learning.

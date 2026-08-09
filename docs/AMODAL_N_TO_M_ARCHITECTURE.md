# Canonical architecture: amodal N-to-M neural computer

This document is the normative architecture specification for the project.
When a design note, experiment README, or historical report uses looser
language, this document controls the meaning of the target system.

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
goal state; it does not store a task-specific policy and does not resize the
controller when candidate count changes.

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

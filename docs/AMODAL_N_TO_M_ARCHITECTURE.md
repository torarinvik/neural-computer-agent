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
access, eviction learned from utility, or cross-adapter retrieval gains.
Memory reads preserve gradients through query-key scoring and value weighting;
inside an explicit differentiable transaction, pending values are mixed by a
trainable write-strength gate. Durable storage mutation remains detached and
explicitly stateful so persistence never captures an autograd graph.
The canonical executable-artifact store additionally exposes verified top-k
opaque candidate promotion. This permits a caller to measure or compose
reusable learned factors without adding task semantics to the memory backend;
single-artifact execution, multi-artifact composition, and transfer policy
remain caller-owned and independently replaceable.
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

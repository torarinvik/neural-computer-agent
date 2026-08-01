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

The original prototype path is:

```text
RGB frame -> VisionEventEncoder inside UnifiedCognitiveController
          -> recurrent controller/memory -> latent intention
          -> actuator inside the same model -> two action logits
```

The first behavior-preserving extraction rung now also provides:

```text
external VisionEventEncoder -> AmodalEvent -> controller.step_event()
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

The cardinality extension passes with the same two agreement seeds. With two
useful streams and up to three valid distractors, N=3 reaches 95.90–95.98%,
N=4 reaches 94.03–94.32%, and N=5 reaches 90.65–91.41%, versus 58.29%,
56.01%, and 53.43% without agreement; N=2 remains 96.61%. The router
therefore has an audited N=2→N=5 path, while cross-modal agreement and more
than two useful streams remain open.

Timestamp-preserving transport alignment is now also qualified. At 4,096
lifetimes, out-of-order delivery and 0.25-unit timestamp jitter reproduced
synchronous actions exactly at 96.36%; genuinely mismatched timestamps stayed
in separate windows. This is transport plumbing, not learned delay handling.
Two outcome-only attempts to fine-tune the input bus against erased pixels
preserved clean accuracy but worsened held-out corruption accuracy, so the
frozen bus remains promoted. Robust noisy-stream handling now requires a
corruption-aware frontend or uncertainty mechanism with its own causal audit.

This is still **not yet the target architecture** because:

- synchronous event sets have been behaviorally qualified at N=1, N=2, and a
  confidence-gated N=3 distractor control;
- learned delay compensation, cross-modality relevance, and fully learned
  missing-stream policies are not yet qualified;
- the recurrent compatibility API still feeds canonical action IDs back into
  the controller, so a new physical protocol requires a thin lowering for
  closed-loop use;
- migration-v1 retains two reserved compatibility coordinates for loading old
  checkpoints, although they are structurally zero in the promoted successor;
- no audio or trained language frontend/backend has passed causal audits.

Therefore current results establish extracted vision-grounded neural IR,
cognition, memory, audited M-output fan-out, and synchronous complementary N=2
composition—not yet unrestricted asynchronous amodal operation.

## Behavior-preserving migration order

1. ~~Extract the existing vision encoder and actuator from controller ownership.~~
2. ~~Add `encode`, `step_event`, and `decode` boundaries.~~
3. ~~Prove bit-identical behavior and checkpoint conversion for the current path.~~
4. ~~Version the event and intention schemas and save adapters separately.~~
5. ~~Remove the active legacy action residual by folding it into the base
   intention without losing any admitted capability.~~
6. ~~Add a second output adapter and prove simultaneous M-output fan-out from
   the clean base intention.~~
7. ~~Accept variable-size synchronous visual event sets.~~
8. ~~Qualify timestamp-preserving out-of-order and bounded-jitter delivery.~~
   Then qualify learned delay, noisy, and missing-stream policies.
9. Add a second synthetic encoder with redundant evidence and measure learning
   acceleration, dropout robustness, and shuffle sensitivity.
10. Require complementary evidence split across two encoders.
11. Train audio and language frontends/backends only after the generic buses pass.
12. Freeze the controller and qualify genuinely new sensors and outputs.

No capability claim may skip from the current internal vision module directly
to “amodal.” Each boundary must pass its own causal and sample-efficiency gate.

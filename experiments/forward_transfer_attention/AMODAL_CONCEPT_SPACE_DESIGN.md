# Emergent amodal concept space

The canonical system-level interface is defined in
[`../../docs/AMODAL_N_TO_M_ARCHITECTURE.md`](../../docs/AMODAL_N_TO_M_ARCHITECTURE.md).
This file supplies representational motivation; the canonical document controls
module boundaries and current-versus-target claims.

## Objective

The agent's native internal output should be a learned, modality-independent
representation of reusable meaning. Text, speech, images, video, and actions
are interfaces that encode into or decode from that representation; none is
the canonical form.

For a concept such as a bird, the central state is not the token `"bird"` or a
stored picture. It is a latent state that can support recognition, retrieval,
prediction, speech, text, imagery, and action. For this project, the same
principle applies first to primitives such as identity, order, selection,
mapping, relation, and rule.

## Architectural constraint

```text
encoder 1 ─┐
encoder 2 ─┼─> variable-size amodal event bus ─> one controller/memory
...       ─┤                                      └─> intention bus ─┬─> decoder 1
encoder N ─┘                                                          ├─> ...
                                                                      └─> decoder M
```

The shared state is the agent's native representational currency. Environment
actions are still emitted through an action decoder, because games require
concrete actions, but the decoder should consume the shared latent rather than
each sensory encoder producing task-specific action logits independently.

## Device-independent intention layer

Action meaning must also be separated from its transport or actuator. In Pong,
the agent should form a latent intention corresponding to moving the paddle up
or down. It should not internally reason in keyboard scan codes, JSON strings,
bit positions, controller buttons, or spoken words.

```text
shared concept/workspace state
              |
       latent intention
              |
     actuator/protocol adapter
              |
   ┌──────────┼───────────┬──────────┬──────────┐
keystroke   JSON      bit flags   speech     robot/game API
```

The actuator adapter may be deterministic when the target protocol is already
known, or learned through calibration when the relationship between commands
and physical effects is unknown. In either case, protocol details stay outside
the reasoner's native action representation.

For example, one learned intention could be rendered as:

- a keyboard `ArrowUp` event;
- `{"direction": "up"}`;
- bit flag `0b0001`;
- the spoken word “up”;
- a target paddle velocity;
- a robot motor command.

These are not six concepts. They are six realizations of one intention.
Likewise, parsing a JSON field or spoken instruction should recover a compatible
latent intention without making JSON or language the internal ontology.

The clean interface is therefore:

`sensory concepts → reasoning/workspace → latent intention → device adapter`

For continuous control, the intention may represent desired effect or change
(`move upward`, `reduce distance`, `intercept trajectory`) rather than a
discrete class. This makes it transferable to actuators with different command
sets, speeds, and dynamics.

The latent should retain two distinguishable kinds of information:

- invariant conceptual content reusable across modalities and instances;
- contextual or instance detail needed for a particular prediction or output.

This need not be implemented as two hard-coded semantic vectors. Learned slots,
factorized subspaces, distributions, or recurrent workspace states are all
valid if causal audits show the required behavior.

## Bitter-lesson constraint

Do not assign human meanings to latent coordinates, require English tokens as
the internal language, provide privileged task state, or train the deployed
agent with human-authored semantic labels. A classical supervised concept
bottleneck is not the target architecture.

Concepts should emerge because sharing a representation improves verified task
performance, sample efficiency, compression, prediction, and transfer.
Deterministic generator/verifier facts may train temporary diagnostic probes
that tell us what information is present and where it is lost. These probes are
scientific measuring instruments only: their weights, labels, and semantic
heads do not enter the deployed agent or count as learned capability.

The deployed learning signals are restricted to experience-derived quantities:

- sensory prediction and cross-view consistency;
- observed action effects and externally verified success/failure;
- memory reconstruction, retrieval utility, and retention;
- compression that preserves verified capability;
- measured improvement and transfer to subsequent tasks.

Even automatically generated semantic answers are excluded from deployed
representation training when they directly reveal the intended abstraction.
The verifier may judge behavior, but it may not teach the latent vocabulary.

## Training pressures

Candidate task-agnostic pressures include:

1. **Cross-modal agreement:** matched visual, auditory, and textual experiences
   should produce states that are interchangeable for downstream behavior.
2. **Modality dropout:** randomly remove available modalities so no single
   representation becomes mandatory.
3. **Cross-modal prediction:** information encoded from one modality should
   predict verified observations or outcomes in another.
4. **Memory reuse:** a state written from one modality should be retrievable and
   useful when queried through another.
5. **Compression without degradation:** reward smaller memory only when
   retention and future-learning audits remain intact.
6. **Behavioral grounding:** verifier reward remains sovereign; representational
   similarity alone is never treated as proof of meaning.

## Required audits

An amodal claim requires more than a decoder producing the right label.

- **Held-out modality transfer:** learn through one modality and solve through
  another without paired fine-tuning on the held-out path.
- **Cross-decoder substitution:** the same frozen latent must drive multiple
  decoders successfully.
- **Actuator substitution:** after minimal adapter calibration, the same frozen
  intention policy must control different command protocols without retraining
  the reasoner.
- **Protocol permutation:** permuting keyboard keys, bit assignments, or API
  action IDs must require changing only the adapter, not relearning Pong.
- **Effect grounding:** when an actuator's dynamics are unknown, the adapter
  must learn commands from observed effects while the intention space remains
  stable.
- **Latent causal swap:** swapping concept states between otherwise matched
  episodes must swap the corresponding verified behavior while preserving
  unrelated context.
- **Modality removal:** performance must survive removal of the modality that
  was easiest during training.
- **Novel-instance generalization:** new appearances, voices, render seeds, and
  symbol identities must preserve the abstract relation.
- **Shortcut controls:** shuffled pairing, garbage memory, stale concepts, and
  mismatched modalities must degrade performance predictably.
- **Fresh-agent comparison:** prior cross-modal concepts must reduce
  examples-to-threshold on later tasks.

## Near-term implementation

The current `UnifiedCognitiveController` still owns its vision encoder and
actuator and accepts one RGB frame per step. Separate Python submodules are not
yet proof of amodal modularity. The first required change is a behavior-
preserving extraction into independently checkpointed event and intention
interfaces; only then should cross-modal experiments begin.

Do not add image, speech, or text generation heads yet. The current action
decoder is enough to test the central scientific claim cheaply.

The next curriculum should train the temporal relation across diverse visual
identities, hold entire palettes out, and measure:

1. rule decodability in the shared write state;
2. behavior on held-out identities;
3. examples-to-threshold versus a fresh agent;
4. retention of earlier mappings, spatial, and shape primitives.

After visual identity invariance works, render the same deterministic primitive
through a second sensory channel. Success means memory written from one channel
can guide actions when the query arrives through the other. This is the
smallest rigorous test of an emergent amodal primitive.

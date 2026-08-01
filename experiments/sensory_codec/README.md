# Sensory codec v0: primitive-game suite

> **Historical experiment:** The streamer/listener arrangement below tests one
> component hypothesis; it is not the target system architecture. See the
> [canonical amodal N-to-M specification](../../docs/AMODAL_N_TO_M_ARCHITECTURE.md).

This isolated experiment tests whether recurrent sensory streamers learn a compact,
temporally useful world representation, rather than merely a private action code.
It does **not** feed learned claims into Screenwatch's trusted ledger.

## Experimental structure

Ten deterministic environments share the actions `UP / RIGHT / DOWN / LEFT`:

| Game | Capability under test |
|---|---|
| `snake` | persistent body state and self-collision |
| `collect` | static obstacles and path planning |
| `dodge` | moving hazards and predicted collision |
| `chase` | target velocity and interception |
| `maze` | procedural layout and route planning |
| `keydoor` | ordered subgoals and visible text state |
| `memory` | delayed target recall after disappearance |
| `patrol` | active pursuer avoidance |
| `signal` | nonvisual audio/text direction commands |
| `rhythm` | audio timing and gated movement |

Targets flicker and game-specific detail is occasionally hidden, so recent frames can
matter. Symbolic teachers supply reproducible action labels and exact simulator state
supplies shared grounding questions.

Three objectives are available:

- `gameplay`: action imitation only; expected to encourage an action shorthand.
- `grounded`: action plus target position, direction, collision, and recent-event QA.
- `compressed`: the grounded objective plus a charge on the streamer's soft gates.

Five listener controls are available:

- `grounded`: pretrain a small listener on privileged symbolic state, freeze it, then
  train the pixel streamer to communicate through its 16-dimensional input.
- `random`: freeze the same randomly initialized listener. Good performance here is
  evidence that the streamer can steer arbitrary fixed logits, not comprehension.
- `direct`: replace the listener with ordinary trainable task heads. This is the
  essential efficient-controller baseline.
- `llm`: freeze a pretrained SmolVLM2 language core behind the three tagged soft-token
  segments.
- `llm_random`: randomly reinitialize and freeze the same language architecture.

The small controls remain essential even when an open-weight listener is enabled: a
large listener is useful only if it beats matched controls on grounding and rollout
tests, not merely on an imbalanced offline action label.

## Sensory firewall

Runtime inference obeys a hard interface implemented in `runtime.py`:

```text
raw pixels ─→ vision streamer ─→ <vision> soft tokens </vision>
raw PCM    ─→ audio streamer  ─→ <audio> soft tokens </audio>
visible characters → text streamer → <text> soft tokens </text>
                                      ↓
                               frozen listener
                                      ↓
                              one action token
```

The three raw channels are synchronized in a typed `SensoryPacket`. Text means visible
HUD character cells (the synthetic analogue of OCR), never simulator narration. Audio
is a rendered waveform, never an event label. The listener receives no game object,
game name, exact state, reward, teacher action,
collision flag, or environment callback. The environment receives only one of
`<UP>`, `<RIGHT>`, `<DOWN>`, or `<LEFT>`. Privileged simulator state may author labels
and score results outside the runtime boundary, exactly as labels may describe a real
video without being pixels in that video.

The `grounded` toy listener remains a diagnostic control pretrained on the shared
semantic vector before it is frozen; it is not the real LLM path. The `llm` listener
uses a pretrained, frozen open-weight language core and is never shown privileged game
state. Only the streamers/projectors see gradients derived from teacher labels, and the
LLM's inference input remains the three tagged sensory representations. Tests reject
mappings, extra channels, and non-`SensoryPacket` inputs at the runtime boundary.

## Run

Use the repository's existing VLM environment:

```sh
./.venv-vlm/bin/python -m unittest experiments.sensory_codec.test_codec

./.venv-vlm/bin/python -m experiments.sensory_codec.train \
  --variant grounded --listener grounded
```

The default trains and evaluates the mixed ten-game suite. Reproduce the original
Snake-only rung with `--games snake`.

The scientifically important run holds out an entire game:

```sh
./.venv-vlm/bin/python -m experiments.sensory_codec.train \
  --variant compressed --listener grounded \
  --holdout-game chase
```

Repeat with each game held out. New seeds within a familiar game measure robustness;
successful held-out games measure representation transfer.

The stricter exam holds out every game that teaches a capability:

```sh
./.venv-vlm/bin/python -m experiments.sensory_codec.train \
  --variant compressed --listener grounded \
  --holdout-capability audio_command

./.venv-vlm/bin/python -m experiments.sensory_codec.capability_matrix \
  --seeds 7,17,29 --jobs 3
```

Capability families currently cover audio commands, text state, persistent memory,
velocity prediction, and route planning. Each run also executes raw-modality traps for
corruption, stale history, missing channels, minimal changes, and uncertainty response.

Run the central comparison:

```sh
for variant in gameplay grounded compressed; do
  for listener in grounded random direct; do
    ./.venv-vlm/bin/python -m experiments.sensory_codec.train \
      --variant "$variant" --listener "$listener"
  done
done
```

For a quick smoke run:

```sh
./.venv-vlm/bin/python -m experiments.sensory_codec.train \
  --samples 512 --test-samples 128 --epochs 2 --listener-epochs 2 \
  --variant grounded --listener grounded
```

Checkpoints and JSON results default to `/tmp/sensory_codec`, keeping generated
artifacts out of the repository. Each result includes task accuracy, average gate
usage, and distances between normal codes and codes from reversed/frozen histories.
The temporal distances are diagnostics, not proof of correct temporal grounding;
later rungs should add explicit temporal questions and counterfactual scoring.

See `RESULTS.md` and `matrix_summary.json` for the three-seed holdout/control matrix,
DAgger comparison, and the first frozen-LLM sensory-firewall pilot.

Run the locally cached frozen-LLM listener with:

```sh
./.venv-vlm/bin/python -m experiments.sensory_codec.train \
  --variant compressed --listener llm --local-files-only \
  --holdout-game chase --batch-size 16
```

Use `--listener llm_random` for the matched randomly initialized frozen-transformer
control. In both cases the runtime boundary remains pixels-to-action-token only.

SmolVLM2 can also supply the frozen visual backbone while the rest of the multimodal
streamer remains trainable:

```sh
./.venv-vlm/bin/python -m experiments.sensory_codec.train \
  --variant compressed --listener llm \
  --llm-model HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --vision-streamer smol \
  --vision-model HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --vision-size 128 --local-files-only \
  --holdout-capability velocity_prediction --batch-size 16
```

Add `--randomize-vision` for the matched randomly initialized frozen vision-tower
control. Result filenames include listener checkpoint, vision backend, and randomized
status so matrix runs cannot silently overwrite one another.

## Accuracy-first latency reward

The default latency weight is deliberately tiny (`--latency-weight 0.002`). During
supervised learning it penalizes routing activity as a differentiable compute-cost
surrogate. Real wall-clock time is noisy and cannot supply useful gradients, so every
run also measures synchronized end-to-end p50/p95 latency from raw `SensoryPacket` to
action token. Model selection reports:

```text
accuracy_dominant_score = action_accuracy
                        - 0.002 × log(1 + mean_latency_ms / 50ms)
```

At a 50 ms latency this subtracts only 0.0014 accuracy points. A materially more
accurate model therefore always wins; latency distinguishes near-ties. Once hard token
or modality routing is introduced, RL can use measured per-action latency directly.

## Initial rung result

A single-seed run on 6,000 teacher samples (12 streamer epochs, 20 listener
epochs) established that the setup is capable of falsifying weak versions of the
idea. The machine-readable summary is in `initial_results.json`. These are
preliminary development measurements, not a general result:

| Objective / listener | Action | Apple-horizontal QA | Danger QA | Apples/episode |
|---|---:|---:|---:|---:|
| gameplay / grounded | 0.737 | 0.827 | 0.756 | 1.23 |
| grounded / grounded | 0.781 | 0.943 | 0.871 | 2.93 |
| compressed / grounded | **0.783** | 0.932 | **0.876** | **3.50** |
| grounded / direct head | 0.736 | 0.903 | 0.765 | 0.87 |
| grounded / random listener | 0.222 | 0.452 | 0.481 | 0.07 |

The symbolic teacher averaged 27.1 apples and the random policy 0.0 under the
same 30-episode audit. Thus grounding supervision and the pretrained listener
both helped, and the small compression charge did not damage accuracy. However,
the best learned policy still died in every episode and achieved only 3.5 apples
on average. The correct conclusion is **promising rung, not learned sensory
language yet**. Repeat across seeds and add explicit temporal/counterfactual
tasks before advancing to an open-weight LLM listener.

## First cross-game smoke result

The leave-`chase`-out path was exercised with a deliberately small development run:
1,600 examples from Snake, Collect, and Dodge; 400 unseen Chase examples; five
streamer epochs. It reached 82.3% horizontal-target QA and survived 244.7 of 250
steps on average in Chase, versus 34.8 for a random policy. However it intercepted
only 2.27 targets versus the teacher's 26.37 and action imitation was 29.5%.
The concise machine-readable record is `crossgame_smoke_results.json`.

This validates the held-out-game machinery and shows limited transfer, not success.
Boundary avoidance and coarse target location transferred more readily than
velocity-based interception—the distinction the wider suite is intended to expose.

## Admission rule

Do not call the representation a sensory language based on game score. It should at
minimum beat the random-listener control on held-out grounding questions, respond to
relevant counterfactual changes, show temporal-corruption sensitivity, and transfer
to another listener with a small adapter. Until then its outputs remain experimental
INFERRED records with source-frame evidence handles.

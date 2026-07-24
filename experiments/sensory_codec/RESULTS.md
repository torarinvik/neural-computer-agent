# Sensory codec results

## Three-seed leave-one-game-out matrix

Each model trained on three games and was evaluated on the fourth. Event transfer is
the fraction of the random-to-teacher rollout gap closed by the learned controller.
It is more stringent than offline action imitation because errors compound during play.

| Held out | Gameplay only | Grounded+compressed | Direct head | Grounded+compressed + DAgger |
|---|---:|---:|---:|---:|
| Snake | 1.0% | **1.7%** | 1.4% | 1.5% |
| Collect | 3.4% | 4.7% | 0.3% | **8.7%** |
| Dodge | 4.4% | 5.3% | 4.7% | **8.7%** |
| Chase | 0.4% | **1.1%** | -0.4% | 0.2% |

Grounding supervision helps consistently but modestly. The grounded listener is much
better than a direct head on Collect, roughly tied on Dodge, and neither solves Chase.
Two rounds of teacher-labelled on-policy aggregation nearly double Collect and Dodge
transfer, while leaving Snake and Chase unchanged. Those two require better persistent
state and velocity representations, not merely more recovery states.

Offline held-out action accuracy for grounded+compressed was 72.2% Snake, 73.0%
Collect, 80.4% Dodge, and 32.2% Chase. Its corresponding event transfer was only
1.7%, 4.7%, 5.3%, and 1.1%. Offline imitation accuracy is therefore not an adequate
success metric for this experiment.

## Frozen-LLM sensory-firewall pilot

The 256M SmolVLM2 language core was used as a frozen LLM. Its vision tower was removed.
At runtime it received four learned sensory embeddings and a fixed generic action prompt;
it emitted one constrained action token. It never received game identity, state, reward,
teacher action, or environment callbacks.

A same-architecture listener with randomly reinitialized frozen weights controls for a
projector steering an arbitrary nonlinear network. The first deliberately small run used
one seed, 2,000 training examples, and five epochs:

| Held out | Listener | Steps / 250 | Events |
|---|---|---:|---:|
| Chase | pretrained frozen LLM | **241.7** | **1.13** |
| Chase | randomized frozen LLM | 26.5 | 0.33 |
| Collect | pretrained frozen LLM | **47.4** | **0.33** |
| Collect | randomized frozen LLM | 4.1 | 0.13 |

The pretrained LLM has a clear stability advantage over the matched randomized control
in this pilot. This is encouraging but not sufficient: it is one seed, interception and
collection remain poor, and soft-prompt steering is still possible. The next admission
exam is a three-seed pretrained/randomized comparison on all four holdouts, followed by
discrete-token and listener-swap tests.

## Ten-game multimodal rung

The suite now includes Maze, KeyDoor, Memory, Patrol, Signal, and Rhythm in addition to
the original four games. Every symbolic teacher was separately audited for 100 steps;
all survived and completed at least one event. This caught and fixed two oracle bugs
before the model results were admitted.

The first all-game development run used independent vision, audio, and visible-character
streamers. Held-in test action accuracy was 86.4%. More importantly, raw-modality
ablations changed the relevant tasks selectively:

| Game | Full action | No vision | No audio | No text |
|---|---:|---:|---:|---:|
| Signal | 88.9% | 87.5% | **48.6%** | 87.5% |
| Rhythm | 94.4% | 67.4% | **78.5%** | 97.2% |
| KeyDoor | 98.6% | 50.0% | 98.6% | **96.5%** |
| Memory | 88.2% | **13.9%** | 88.2% | 88.2% |

Signal primarily learned the waveform direction code; removing text affected its
horizontal grounding more than action because audio carried a redundant command. Rhythm
used both vision and audio. KeyDoor's action was only mildly text-dependent because the
key itself remained visually observable, so a later trap should make the same visual
state ambiguous without the `KEY HELD` surface. These ablations demonstrate channel use,
not cross-game generalization; ten-game leave-one-capability-out evaluation remains next.

## Latency objective

Training now includes a default 0.002 routing-cost weight, while evaluation measures
synchronized raw-sensors-to-action-token latency. In a small same-seed smoke comparison,
accuracy was identical with and without the latency term; average routing gate activity
fell from 0.4853 to 0.4833. Measured mean MPS latency was 3.44 ms versus 2.75 ms, but a
single short timing run is too noisy to claim a speedup. The important verified property
is priority: the resulting latency penalty was only 0.00011 accuracy points.

## Capability-family holdouts and modality traps

A three-seed development matrix removed every game teaching each capability before
testing that family. This is much stricter than holding out one visual environment:

| Capability held out | Action | Event transfer | Corruption | Stale | Missing entropy Δ |
|---|---:|---:|---:|---:|---:|
| Audio command | 51.5% | **0.0%** | 33.3% | 29.2% | -0.033 |
| Text state | 57.6% | **0.4%** | 25.0% | 25.0% | -0.020 |
| Persistent memory | 62.2% | **0.5%** | 70.8% | 95.8% | +0.679 |
| Velocity prediction | 70.2% | **4.6%** | 75.0% | 100% | +0.279 |
| Route planning | 79.4% | **27.8%** | 75.0% | 100% | +0.361 |

The route-planning mean is misleadingly positive: Maze transferred at 75%, while
Collect reached 4.2% and KeyDoor 2.2%. Likewise, velocity transfer was mostly Dodge
(15%); Chase was negative and Patrol zero. No capability family passes admission.

When audio-command games were entirely absent, both the pretrained frozen LLM and its
matched randomized control scored 51.4% offline action accuracy, chance-level 25% on
the Signal modality traps, and zero useful closed-loop transfer. Language pretraining
did not supply unseen waveform semantics. Worse, missing audio/text sometimes reduced
entropy, meaning the system became confidently wrong. The next training rung needs an
explicit cross-modal alignment and uncertainty objective, not merely more game imitation.

## SmolVLM2-500M placement and control pilot

SmolVLM2-500M was tested in both plausible locations: as a frozen visual encoder and
as the frozen tagged-soft-token listener. This was a single-seed velocity-capability
holdout pilot, so it selects the next experiment rather than establishing a result.

With the same grounded diagnostic listener and 1,600 training samples:

| Vision encoder | Action | Zero vision | Target-erasure code distance | Mean latency |
|---|---:|---:|---:|---:|
| tiny trainable | **70.8%** | 16.2% | **0.0858** | **2.75 ms** |
| SmolVLM2-500M pretrained, frozen | 62.0% | 26.2% | 0.0400 | 19.49 ms |
| SmolVLM2-500M randomized, frozen | 70.5% | **70.5%** | 0.000005 | 20.93 ms |

The pretrained tower did react to the grid, unlike its randomized control, but it was
less accurate and about seven times slower than the tiny encoder. This is unsurprising:
12×12 symbolic grids enlarged with nearest-neighbour sampling are far outside a natural
image/video tower's training distribution. The randomized tower's apparent 70.5% is a
particularly useful trap result: identical zero-vision accuracy and near-zero target
sensitivity reveal a label-prior shortcut.

With the tiny visual encoder and 1,000 training samples, the pretrained 500M listener
beat its randomized twin (70.8% versus 64.0%) at essentially identical latency (24.21
versus 24.25 ms). However, zeroing vision changed neither score, target erasure barely
moved either representation, and both closed only 2.8% of the rollout event gap. The
pretraining advantage is real in this seed but is not grounded sensory understanding.

The current default should therefore remain the tiny trainable visual streamer, with
SmolVLM2-500M retained as the preferred pretrained-listener research rung. Before any
larger sweep, add balanced/counterfactual action scoring and a loss that forces current
visual evidence to matter. Machine-readable measurements are in
`smolvlm2_results.json`.

### Ten-seed RTX 5090 follow-up

The frozen 500M listener comparison was expanded to ten paired seeds on an RTX 5090.
Every run used 4,000 training examples, 1,000 held-out examples, eight epochs, and
the entire velocity-prediction family as the capability holdout:

| Frozen listener | Action | Zero vision | Event transfer | Target-erasure distance | Latency |
|---|---:|---:|---:|---:|---:|
| SmolVLM2-500M pretrained | 60.9% ± 9.8% | **35.2%** | **2.50% ± 2.02%** | **0.2442** | 11.51 ms |
| same architecture, randomized | **69.2% ± 6.6%** | 63.0% | 1.57% ± 0.63% | 0.0099 | 11.47 ms |

The paired pretrained-minus-randomized action difference was -8.35 percentage
points; pretrained won only three of ten seeds. Its rollout-transfer advantage was
+0.92 points and it won five seeds, which is too small and inconsistent to count as
successful generalization. The important positive result is causal rather than
behavioral: pretraining greatly increased visual dependence and target sensitivity.
The important negative result is that this extra sensory information did not become
competent action. Balanced counterfactual training is therefore a prerequisite for
the next model-size sweep.

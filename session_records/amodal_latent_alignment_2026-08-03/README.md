# Latent-basis alignment through paired sensory consistency (2026-08-03)

## Question

Can a new encoder with a different latent coordinate system become useful to a
frozen controller without semantic labels, task IDs, correct-action labels, or
verifier outcomes? The diagnostic wraps a copied vision frontend in a fixed
reverse permutation of its latent coordinates. A trainable linear adapter must
recover the controller's existing basis. The controller, original encoder,
input bus, and decoder remain frozen; only the second encoder's adapter is
trainable.

## Results

The self-supervised signal was paired encoded-event consistency: for the same
unlabeled raw frame, the new stream's adapted encoding is pulled toward the
original stream's encoding. No semantic or verifier bits are used. Each run
used 32 updates, batch size 256, all three appearances (`bars`, `diamonds`,
`dot_pairs`), and 147,456 unlabeled calibration frames. Two identity-initialized
seeds and one randomly initialized adapter passed the pre-registered gates:

| seed | bars fused | diamonds fused | dot-pairs fused | controller changed |
|---:|---:|---:|---:|:---:|
| 2871 | 96.04% | 96.27% | 95.90% | no |
| 2872 | 96.07% | 96.43% | 95.98% | no |
| 2873 (random init) | 98.09% | 95.31% | 93.93% | no |

The individual streams stayed near chance, shuffled partners stayed near
chance, and contradictory partners caused the expected prediction reversals
(approximately 80--93% across appearances). Full N=1 behavior remained about
98--100%, within the retention gate. The random-init curve crossed the useful
composition range around update 8, rather than relying on the adapter's
identity initialization. The reports contain the complete curves and hashes.

This qualifies a narrow but important property: **paired, unlabeled sensory
consistency can align a new neural-IR basis while the sovereign controller is
frozen, and the aligned stream participates in an already learned
cross-stream relation.** It is not a claim that arbitrary natural audio,
language, or a cold-start encoder will align without paired correspondence.

## Reward-only controls

Three reward-only arms were deliberately kept as negative evidence. They
briefly improved fused accuracy on their small training stream, but failed
shuffled/contradictory controls and did not cross the pass gate:

| arm | final fused | shuffled partner | contradictory flip |
|---|---:|---:|---:|
| bus + adapter, seed 285001 | 79.38% | 53.01% | 55.82% |
| bus + adapter, seed 285002 | 75.90% | 51.09% | 53.67% |
| adapter only, seed 285003 | 71.64% | 58.44% | 39.06% |

This is evidence against promoting sparse reward as the first alignment
signal—not evidence that reward learning is impossible at a larger, properly
curriculated scale. It is also a reminder that a high fused score without a
causal contradiction audit is not sufficient.

## Sample-efficiency interpretation

The alignment run used no verifier bits and learned from a dense, automatically
available relation between two views of the same sensory event. That is a
useful sample-efficiency result, but its unit is unlabeled paired frames, not
reward episodes. The next comparison must measure the cost of acquiring those
pairs and test whether the adapter can align a genuinely different raw
modality under the same no-semantic-label rule.

## Artifacts and next gate

- `self_supervised_seed2871.json`
- `self_supervised_seed2872.json`
- `self_supervised_random_seed2873.json`
- `reward_only_bus_adapter_seed285001_u4.json`
- `reward_only_bus_adapter_seed285002_u4.json`
- `reward_only_adapter_only_seed285003_u16.json`
- controller: `artifacts/checkpoints/unified_repertoire_span2_amodal_intention_seed122005.pt`
- input bus: `artifacts/checkpoints/amodal_input_bus_complementary_seed145001.pt`

No production checkpoint is promoted from this diagnostic. The next highest-ROI
step is to save the adapter as an independently loadable artifact, replay the
frozen-controller composition audit from that artifact, and then repeat the
paired-consistency alignment with a non-vision encoder (audio or token stream)
before allowing any controller or memory updates.

## Artifact replay and audio frontier

The visual adapter was then saved independently as
`artifacts/checkpoints/amodal_latent_basis_adapter_seed2874.pt`. A separate
process loaded that artifact—without the training optimizer—and passed a fresh
2,048-lifetime replay audit. Bars/diamonds/dot-pairs fused accuracy was
96.39%/96.81%/96.12%; shuffled partners stayed at 51.19%/53.45%/49.32%, and
contradictory prediction flips were 87.63%/81.64%/93.44%. This proves the
result is reusable checkpoint state rather than a training-process effect.

The first genuinely different raw modality was audio. A deterministic sensor
rendered each ordinary visual view as a waveform whose carrier amplitudes were
the low-resolution RGB measurements. A fixed spectral demodulator recovered
raw sensor features, while only a small trainable projection learned the
vision encoder's opaque neural-IR basis. The controller, vision encoder, input
bus, and decoder remained frozen. Training used 16 updates, batch size 128,
three appearances, two paired views per trial, and **73,728 paired frames**;
there were zero verifier bits and no semantic labels.

An independent 2,048-lifetime replay of the saved audio frontend passed:

| appearance | fused | shuffled partner | contradictory flip |
|---|---:|---:|---:|
| bars | 95.21% | 52.10% | 84.85% |
| diamonds | 88.96% | 53.29% | 69.50% |
| dot-pairs | 94.60% | 49.37% | 89.93% |

N=1 retention stayed 97.83--99.98%, and both individual streams remained near
chance. The promoted adapter is
`artifacts/checkpoints/amodal_audio_aligned_frontend_seed289001.pt`; the
training and replay reports are `audio_alignment_seed289001.json` and
`audio_adapter_replay_seed989001.json` in this directory.

The first convolutional waveform frontend is retained as
`audio_alignment_conv_failed_seed289001.json`: its loss decreased, but it
failed the causal gates. The successful spectral demodulator therefore records
a useful lesson for sample efficiency: improve the raw representation's
recoverability before spending controller compute. Natural audio, arbitrary
token streams, asynchronous audio, and cold-start cross-modal alignment remain
open.

## Audio corruption and timing audit

The saved audio frontend was stress-tested on two fresh 512-lifetime seeds. It
retained the clean composition gate and remained causal under Gaussian waveform
noise up to one signal RMS and 25% contiguous burst dropout:

| seed | clean minimum | Gaussian 0.50 minimum | burst dropout minimum |
|---:|---:|---:|---:|
| 990002 | 88.55% | 88.05% | 88.63% |
| 990003 | 89.22% | 88.67% | 88.75% |

The audit also includes sample dropout from 10--50%, full audio omission,
low-confidence audio, and a one-trial audio delay. The first two corruption
families preserve the useful stream; omission, delayed evidence, and low
confidence correctly collapse toward the single-view/chance baseline. This
localizes the remaining frontier to a learned missing-event and timing policy,
not denoising. Evidence is in
`audio_robustness_seed990002.json` and `audio_robustness_seed990003.json`.

## Timestamp-buffer breakthrough

The remaining one-trial delay failure was repaired at the transport boundary,
not by adding task logic to the controller. `AmodalEventWindowBuffer` holds
opaque events by their generic timestamps and releases a complete window when
all registered stream handles have arrived. It never reads payload semantics,
correct answers, or modality names.

Two fresh 512-lifetime audits show the effect:

| appearance | buffered one-step delay | out-of-order buffered | unbuffered delay |
|---|---:|---:|---:|
| bars | 93.01% | 93.01% | 49.57% |
| diamonds | 89.02% | 89.02% | 52.73% |
| dot-pairs | 95.55% | 95.55% | 49.61% |

The buffered delayed path processes every window with exactly one step of
arrival latency and no pending windows. The synchronous and out-of-order paths
are identical. This qualifies generic timestamp synchronization for a delayed
synthetic audio stream; a learned wait/timeout policy for genuinely missing
events remains open. Evidence is in
`audio_timing_buffer_seed991001.json` and
`audio_timing_buffer_seed991002.json`.

The buffer now supports an explicit bounded timeout. When audio is absent, it
releases a partial collection with the audio slot marked `present=False` and
confidence zero rather than inventing a payload or waiting forever. On a
redundant full-view task, omitting one audio event preserved 89.26--99.49%
accuracy across appearances, released exactly one partial window, and left no
pending windows on both 512-lifetime seeds. On the complementary task, missing
audio remains near chance—as it must because the remaining view lacks the
required evidence—but the timeout still makes progress deterministically.
This is generic bounded missing-event handling, not a task-specific fallback.
Reports: `audio_timing_timeout_seed991001.json` and
`audio_timing_timeout_seed991002.json`.

## Learned wait/proceed timing breakthrough

The next rung kept the controller, encoders, and input bus frozen. A tiny
arrival predictor learned from observed transport traces whether a pending
event would arrive before the declared deadline; it saw only presence, age,
and recent arrival history, never payloads, stream names, task labels, or
answers. Its held-out trace accuracy was 75.68% with Brier score 0.174.

On two independent 128-lifetime audits, an adaptive policy using that predictor
matched the fixed two-step timeout within two percentage points of accuracy and
within 0.01 verified utility for every bars/diamonds/dot-pairs appearance,
while reducing mean query latency by 0.04--0.08 event units. Removing arrival
history lowered verified utility by 0.005--0.018, and an inverted-history
control was also recorded. Redundant full-view behavior stayed above the 85%
gate. This is a learned timing decision, not recovery of information that was
never emitted; genuinely complementary missing evidence remains impossible and
is still an explicit negative control. Reports: `arrival_predictor_training_seed992001.json`,
`adaptive_wait_seed992101.json`, and `adaptive_wait_seed992201.json`.

The threshold is no longer a hand-selected capability result. A verifier-only
grid over six thresholds on a non-held-out split selected **0.15** by pooled
accuracy/latency utility; no semantic labels were supplied to the policy. Two
fresh held-out 128-lifetime audits using that selected threshold both passed
the pooled accuracy, utility, latency, redundant-view, and no-history gates.
The calibration and final reports are `wait_threshold_calibration_seed993001.json`,
`adaptive_wait_calibrated_seed992101.json`, and
`adaptive_wait_calibrated_seed992201.json`.

## Independent token-sensor alignment

The next boundary tested whether a second raw stream could be made compatible
with the same frozen controller without introducing a language-specific
reasoner. A fixed 16x16 raster sensor converted each ordinary RGB view into a
sequence of continuous three-channel patch tokens. A small token frontend was
trained only with paired encoded-event consistency against the frozen vision
encoder. The controller, original vision encoder, input bus, and decoder were
unchanged; training used no verifier bits, semantic labels, token names,
correct-action labels, or task metadata.

Two independent 32-update runs passed the pre-registered composition and
causal gates. A separate 512-lifetime replay loaded the saved frontend without
the training optimizer and reported:

| appearance | fused | shuffled partner | contradictory flip | N=1 |
|---|---:|---:|---:|---:|
| bars | 98.32% | 51.41% | 93.59% | 97.54% |
| diamonds | 91.88% | 54.22% | 75.70% | 99.96% |
| dot-pairs | 94.69% | 50.16% | 90.59% | 97.85% |

The individual streams remained at their single-view baselines. The
cross-episode shuffle control stayed near chance, while the contradictory
partner replay produced the expected relation reversal. This is the first
qualified token-stream bridge into the shared neural IR in addition to the
synthetic audio bridge. It is not yet natural language-token grounding,
arbitrary cold-start modality alignment, or reward-only encoder learning.

Artifacts:

- `artifacts/checkpoints/amodal_token_aligned_frontend_seed996001.pt`
- `artifacts/checkpoints/amodal_token_aligned_frontend_seed996101.pt`
- `token_alignment_seed996001.json`
- `token_alignment_seed996101.json`
- `token_adapter_replay_seed996901.json`

The first two tokenizations (8x8 and 16x16 integer codebooks) reduced
consistency loss but failed the causal gate. The promoted continuous 16x16
sensor fixed that information bottleneck; training on the held-out palette
family was also necessary for shape generalization. This records a useful
sample-efficiency lesson: improve recoverability and variation at the raw
sensor boundary before spending controller updates.

## Sparse timing-reward controls (negative evidence)

The timing branch also tested whether scalar verifier utility could replace
the calibrated/self-supervised arrival target immediately. A small
metadata-only policy-gradient arm, both from scratch and initialized from the
arrival predictor, collapsed toward fixed wait/release behavior rather than
recovering the adaptive policy. A small UCB threshold bandit was less brittle
but selected different thresholds across short seeds and batch sizes. These
are bounded sample-starvation/credit-assignment negatives, not evidence that
reward learning is impossible. The verifier-only threshold calibration at
0.15 remains the promoted timing policy; the experimental trainers are kept
in `train_amodal_wait_reward.py` and `train_amodal_wait_bandit.py` for a later
larger-data study.

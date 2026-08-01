# Dual-stream follow-up and audit corrections — 2026-08-01

This record follows the proven frozen-controller audio result. The question
was whether one controller can combine simultaneous vision and audio streams
through the same external-memory/computation path.

## What the tiny runs established

The dual policy is not dead, but it is asymmetric. With frozen no-label
vision/audio reconstruction encoders, a factorized opaque decoder, a generic
one-step external-memory adapter, and 256 reward-only updates, the exact
held-out decomposition was:

| measure | result |
|---|---:|
| vision bit | **100.0%** |
| audio bit | 55.3% |
| joint exact action | 55.3% |

The visual stream was therefore mastered while the audio stream remained at
the stochastic baseline. Resetting or shuffling history removed the temporal
advantage. This is a useful localization: simultaneous composition is not yet
working, but the failure is concentrated in the second stream rather than a
total collapse of the controller.

The representation audit found a likely interface pressure: the visual
frontend's event vectors have roughly ten times the RMS of the audio frontend
(about 4.0 versus 0.4 per coordinate). The default bus therefore gives the
visual stream a much larger numerical footprint. Optional task-agnostic RMS
normalization and a target-RMS calibration were added as diagnostic controls;
neither showed a causal promotion in 64–128-update runs. Presenting streams as
serial events also did not pass. These are bounded negatives, not evidence
that arbitrary multimodal fusion is impossible.

## Two audit bugs caught before promotion

1. A legacy two-action decoder was being copied into the four-row factorized
   decoder as joint-mask rows. The corrected loader maps a vision/audio source
   into the corresponding factorized bit rows.
2. Positive factorized partial reward is not joint accuracy: one correct bit
   can make the mean reward positive. The evaluator now reports exact action
   accuracy separately from `partial_reward_accuracy`. Re-evaluating the
   per-bit-credit pilot gave vision 100.0%, audio 56.2%, joint 56.2%, so its
   apparent 100% headline was correctly rejected.

## Current boundary and next experiment

The frozen-controller claim remains valid for the audio-only skill (two
replicas at 67.08%/67.07%, with reset and shuffle controls near 57%). The
simultaneous vision+audio claim remains unpromoted. The next high-ROI design
should preserve source identity and per-stream memory at the amodal boundary
(for example, source-key-conditioned or slot-preserving transport), then
repeat the same exact/reset/shuffle/bit-ablation audits. Do not spend a long
reward-only run on the current mean-pooling bus until a small supervised
representation probe shows that both bits are decodable.

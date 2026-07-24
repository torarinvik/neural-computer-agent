# Audiovisual reflex results

Three seeds (7, 17, and 29) used 3,000 training trials and 1,500 held-out
trials each. The SmolVLM2-500M listener remained frozen. Only the sensory
adapter and, in learned mode, its event thresholds were trained.

| Stream | Accuracy | Tokens/trial | Token reduction | Mean latency |
|---|---:|---:|---:|---:|
| Dense | 100% | 24.00 | 0% | 13.63 ms |
| Fixed sparse | 100% | 2.32 | 90.29% | 13.94 ms |
| Reward-trained sparse | 100% | 2.32 | 90.29% | 13.93 ms |

Every individual seed also scored 100% on vision-only, audio-only, combined,
target, and hazard subsets. Thus the aggregate does not hide a dropped modality.

## What worked

- A simple immediate task made representation failures visible within minutes.
- The frozen LLM learned to act from adapter embeddings without direct game hooks.
- Fixed and learned gates removed about nine of every ten input tokens without an
  accuracy penalty.
- Exact expected one-step reward kept accuracy primary while a normalized 0.001
  emission term supplied the small efficiency incentive.

## What failed or disappointed

- The first sampled policy-gradient objective was unstable: on seed 7 it reduced
  accuracy from 87.33% to 74.80% and drove vision-only accuracy to chance by
  learning to suppress visual events. This run was rejected, not averaged into
  the confirmed result.
- The original one-pixel visual cue fell just below the fixed frame-delta
  threshold. A regression-tested 3x3 raw-pixel cue corrected the stimulus.
- A 90.29% token reduction did not improve wall-clock latency. Physical token
  compaction is outweighed by adapter and frozen-transformer overhead at this
  short sequence length.

## Conclusion

The immediate arena validates the core information-bottleneck proof of concept:
a sparse audiovisual streamer can preserve all information needed by a frozen
LLM. It does not yet validate a latency advantage or broad game generalization.
The next experiment should use longer streams, cached/incremental listener state,
and harder held-out cue variants where selecting the right events matters more
than detecting a single obvious onset.

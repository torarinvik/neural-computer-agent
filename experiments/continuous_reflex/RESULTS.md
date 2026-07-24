# Continuous noisy reflex results

Three seeds used 3,000 training windows and 900 held-out windows each. Every
32-tick window contained seven irrelevant visual/audio events. Held-out trials
shifted visual themes, cue appearance, timing, and audio frequency within the
same semantic bands. SmolVLM2-500M remained frozen and received only adapter
embeddings derived from raw pixels and PCM.

| Stream | Held-out accuracy | Tokens/window | Reduction | Relevant recall |
|---|---:|---:|---:|---:|
| Dense | 81.81% | 64.00 | 0% | 100% |
| Fixed change gate | 80.52% | 12.82 | 79.96% | 100% |
| Learned content gate | 72.37% | 35.80 | 44.07% | 69.79% |
| Random budget | 75.67% | 14.41 | 77.49% | 25.11% |

The random control receives 12% more tokens than fixed, making the comparison
conservative in random's favor. Fixed still leads it by 4.85 accuracy points.

## Successes

- The task no longer saturates at 100%; held-out shifts expose real failures.
- Fixed event detection retains 98.4% of dense accuracy while removing about
  four of every five tokens.
- Fixed gating retains every audited relevant event and beats a near-budgeted
  random sampler.
- Visual generalization is strong: dense averages 100% and fixed 99.34%.
- Persistent KV caching works, though this short 16-event listener-only benchmark
  yields only about 1.08x speedup.

## Disappointments

- The learned content gate does not learn relevance reliably. It uses almost
  three times as many tokens as fixed, recalls only 69.8% of relevant events,
  and performs worse than random.
- Joint reward/sparsity training showed the same silence-collapse frontier as
  Snake: weak efficiency weights stay dense; stronger weights discard useful
  information. Fixed warm-up and balanced raw-event anchoring prevent total
  collapse but do not solve selection.
- Audio is the major generalization bottleneck. Mean audio-only accuracy is
  46.11% dense, 50.44% fixed, 29.67% learned, and 29.56% random.
- End-to-end compact latency remains flat (roughly 13.7–14.2 ms) because adapter
  and frozen-listener overhead dominate these short windows.

## Conclusion

The experiment supports fixed sparse streaming, not reward-learned content
selection. The next iteration should preserve the fixed gate, improve audio
representation with frequency/waveform augmentation, and train a relevance
ranker over already-detected events under an explicit token budget. That separates
event detection, representation learning, and budget selection instead of asking
one unstable gate to learn all three simultaneously.

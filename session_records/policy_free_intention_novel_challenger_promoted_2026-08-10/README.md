# Verifier-selected copy-or-fresh admission — 2026-08-10

This audit closes the next generalization safety boundary for frozen-core
continual learning. A mastered adaptive seven-stage evidence sequence is
given an unseen evidence combination and an unseen intention target. The
external router creates two isolated candidates: one copies a protected
source cell and one starts fresh. Each receives the same bounded,
outcome-only probe; only the higher-scoring candidate is deployed.

The direct-copy baseline showed negative transfer on all three seeds: blind
copy adaptation took `63/68/85` updates against fresh `34/26/61` updates.
The verifier challenger rejected the copied prior in every warm and matched
fresh branch, then all six selected fresh candidates mastered the novel task
and passed held-out retention at scores from `0.9802` to `0.9839`. The controller and state adapter stayed frozen,
the source digest stayed unchanged, persistence was exact, and replay was
zero. The novel capability also survived a later noisy reversal stress test.

| seed | warm probe transfer/fresh | fresh probe transfer/fresh | novel score | continuation updates |
| ---: | ---: | ---: | ---: | ---: |
| 85301 | 0.0937 / 0.9342 | 0.0660 / 0.9328 | 0.9805 / 0.9803 | 12 / 12 |
| 85302 | 0.2595 / 0.9799 | 0.0153 / 0.9799 | 0.9839 / 0.9839 | 1 / 1 |
| 85303 | 0.0836 / 0.9561 | 0.1573 / 0.9561 | 0.9817 / 0.9817 | 7 / 7 |

The audit also includes reward-shuffled, action-shuffled, missing-evidence,
memory-corruption, exact-reload, frozen-core, and post-reversal retention
controls. Shuffled branches are judged by sample-efficient acquisition rather
than endpoint score alone, since a random branch can land near a target by
chance. Every promotion gate passes across all three seeds.

Reproduce from the repository root:

```bash
.venv/bin/python experiments/policy_free_intention_routing/novel_challenger.py \
  --seed 85301 \
  --report-out /tmp/policy-free-intention-novel-challenger.json
```

This promotes bounded verifier-selected copy-or-fresh external intention
admission for one unseen evidence combination and target. It does not yet
promote positive transfer on novel tasks, arbitrary new computation,
unrestricted memory growth, learned compression, or general continual
learning.

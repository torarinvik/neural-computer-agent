# Adaptive sequential evidence-version routing — 2026-08-10

This audit replaces the rejected fixed-boundary multi-stage curriculum with
an adaptive stage protocol. The controller and state adapter remain frozen.
Seven observation-mask versions are presented in order; each version must
reach mastery and pass a fresh held-out eight-outcome prefix verifier before
the next protected external cell is created.

Adaptive forks copy factorized external content and the route key on
dimensions already observed. The sole unqualified child receives a `0.75`
exploration floor, which makes new capacity discoverable without passing a
caller-selected cell index. Each stage has a four-update minimum. Warm and
matched-fresh learners use the identical stage policy, masks, verifier, and
exploration configuration.

All three seeds pass stage completion, source/successor retention,
caller-free routing, reversal, shuffled reward/action, missing-evidence,
corruption, persistence, frozen-core, and zero-replay gates.

| seed | successor updates | matched fresh | transfer ratio | unique verifier bits |
| ---: | ---: | ---: | ---: | ---: |
| 85301 | 39 | 50 | 1.2821 | 559 |
| 85302 | 42 | 44 | 1.0476 | 563 |
| 85303 | 34 | 55 | 1.6176 | 573 |

Each run passes seven stage verifiers, uses `128` held-out stage-verifier
bits, grows nine external cells, and replays zero examples.

Reproduce from the repository root:

```bash
.venv/bin/python -m experiments.policy_free_intention_routing.train \
  --seed 85301 \
  --masked-context \
  --mask-curriculum adaptive_versioned_multi_stage \
  --factorized-context-residual \
  --adaptive-stage-min-updates 4 \
  --report-out /tmp/policy-free-intention-adaptive.json
```

This promotes bounded adaptive sequential reuse across this seven-stage
evidence curriculum. Novel mask orderings, unseen evidence combinations,
unrestricted growth, learned compression, arbitrary new computation, and
general continual learning remain unqualified.

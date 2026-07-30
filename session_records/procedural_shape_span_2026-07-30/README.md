# Procedural shape span: first compounding rung

## Result

The controller learned a new sensory-to-memory curriculum using only RGB
events, its own opaque actions, and scalar attempted-action outcomes:

1. identify one of two currently visible procedural shapes;
2. remember one shape and recognize an independently rerendered candidate;
3. remember two sequential shapes and judge candidates for visually cued
   ordinals presented in counterbalanced order.

The generator crosses every identity sequence, answer pattern, and query
permutation. Thus identity, candidate, query time, ordinal, and correct action
are exactly balanced. Matching images are independently rerendered, so exact
pixel matching is unavailable. Even randomness level zero has a deliberate
nonzero nuisance floor: ±1 pixel position, ±1% size, ±1 degree rotation,
colour variation 0.02, and background variation 0.01.

## Sample-efficiency result

| lineage | span-2 stable bits to 90% | final held-out |
|---|---:|---:|
| inherited seed 271xx | 20,480 | 100.00% |
| inherited replica 272xx | 16,384 | 93.75% |
| fresh seed 27103 | not reached by 32,768 | 50.68% |
| shuffled outcomes | not reached by 16,384 | 50.00% |

For the first inherited run, the 20,480 figure is cumulative across its two
span-2 phases: 16,384 bits before continuation plus 4,096 bits to the first
threshold that remained satisfied. Precursor experience is accounted
separately: 2,048 visible-identity bits and 8,192 span-1 bits. The comparison
is the cost of acquiring the *new* span-2 skill from inherited versus fresh
weights. Because fresh training never crossed the gate, the measured transfer
ratio is a lower bound greater than 32,768 / 20,480 = 1.6x.

The first complete inherited lineage took about 28.4 seconds locally on MPS.
No replayed examples were used. Every training batch contained unique logical
lifetimes.

## Best-seed causal audit

- normal span 2: 100.00%
- each ordinal: 100.00%
- blank presentation: 50.39%
- all fast memory reset before query: 50.05%
- workspace disabled: 82.96%
- recurrent state reset while preserving workspace: 75.49%
- valid reversed-presentation rerender: 100.00%
- prediction flips on every answer changed by reversal: 100.00%
- valid candidate-identity counterfactual: 100.00%
- prediction flips under candidate counterfactual: 100.00%
- retained visible-identity primitive: 100.00%
- retained span-1 recognition: 100.00%

The gap between workspace-disabled, recurrent-reset, and full-reset results
shows that both RAM/VRAM-resident carriers contribute and partly compensate
for one another.

The replica crossed the mastery gate and passed missing-evidence/full-reset
controls, but finished below the best seed and had correspondingly imperfect
counterfactual flip rates (75% presentation reversal, 87.5% candidate flip).
It establishes replication of learning, not replication of perfect behavior.

## The early false positive

The first generator accidentally alternated match/no-match labels along the
flattened query axis. With span 2, this made the correct action a deterministic
function of query ordinal. A model reached 100% while blank presentation and
full-memory reset also remained 100%. The run was rejected immediately.

The repair balances the full Cartesian logical design and varies query order
independently of content and answer. Under the repaired design, a fresh model
remains at chance and the learned model requires visual evidence and memory.
This is a permanent regression test.

## Frontier

Do not jump directly to long spans. The next experiment changes one axis:
raise nuisance randomness slightly while keeping span 2 and rehearse the
current floor. Only after retained span-2 accuracy and causal controls pass
should the curriculum increase toward full position/size/rotation/colour
variation. Span 3 follows after nuisance robustness. At every rung compare
inherited and fresh stable bits-to-threshold; that comparison, not final
accuracy, is the compounding-gains score.

Curated checkpoint:

`artifacts/checkpoints/unified_procedural_shape_span2_seed27104.pt`

SHA-256:

`6e06709df15eb6b706ef02e1f2c763c78f3329e25885f28f95a2b0782016a276`

# Complementary amodal input composition — 2026-08-01

## Breakthrough

A frozen cognitive controller now solves a relation task from two simultaneous
encoded sensory events even though neither event contains enough evidence to
solve it alone.

The scene is divided into two ordinary pixel streams along a neutral spatial
boundary. Each stream passes independently through the external vision
frontend. A generic permutation-invariant input bus combines the resulting
events before the unchanged controller sees them. The bus receives no task,
modality, object, relation, or answer labels.

The learned component is a 4,817-parameter set residual. It was trained only
from the agent's own attempted binary action and the scalar success of that
attempt. The controller, vision encoder, decoder, RAM/VRAM machinery, and disk
memory weights remained frozen.

## Exact compatibility invariants

The bus starts at uniform confidence-weighted attention. A diversity gate makes
the learned residual structurally zero when only one event is present or when
all events are identical. These properties remain true after training:

- N=1 exactly reproduces the old event and controller behavior;
- duplicating an event causes zero logit drift and zero action changes;
- reversing event order causes zero logit drift and zero action changes;
- cardinality can vary per batch example through a presence mask; and
- controller parameter shapes do not depend on N.

## Sample efficiency and replication

Three independent 16-unit bus seeds crossed the stable 85% complementary gate
after 768, 1,344, and 1,344 verifier bits. Their held-out bars accuracies were
89.53%, 96.72%, and 93.83%, while the two individual streams remained between
43.44% and 57.34%.

A matched reward-shuffled run stayed at 57.50% and never crossed. This rejects
reward-independent drift and confirms that the outcome signal caused learning.

## Promoted 4,096-lifetime audit

The selected bus was trained on bars only. It passed complementary composition
on bars and transferred without further updates to two unseen renderers:

| Appearance | Stream A | Stream B | Fused N=2 | Shuffled partner |
| --- | ---: | ---: | ---: | ---: |
| Bars | 55.84% | 45.02% | **96.46%** | 51.77% |
| Diamonds | 49.87% | 50.11% | **90.96%** | 52.36% |
| Dot pairs | 51.29% | 49.89% | **95.63%** | 50.14% |

Contradictory partners drove accuracy against the original answer to 9.80%,
17.21%, and 3.99%, with corresponding prediction-flip rates of 86.67%, 73.75%,
and 91.64%. This is causal composition, not a single-stream shortcut. All
parameters were unchanged during audit.

The bars capability replicated across all three training seeds. Strict
cross-appearance graduation did not replicate: the other two buses reached
82–87% on diamonds and 87–89% on dot pairs. Cross-renderer transfer is therefore
a strong selected-seed result, not yet a replicated sample-efficiency claim.

## Rejected paths

- Uniform averaging without a learned set residual reached only about 59% on
  the strict complementary split.
- Equal early mixing of bars, diamonds, and dot pairs prevented bars mastery.
- Interleaved augmentation also failed at the same tiny budget.
- Adding new appearances only after bars ignition caused interference and
  degraded causal reversal. It was not promoted.

The next curriculum experiment should protect the mastered generic composition
while introducing renderer variation gradually, rather than changing the
appearance distribution abruptly.

## Artifacts

- Controller:
  `artifacts/checkpoints/unified_repertoire_span2_amodal_intention_seed122005.pt`
- Input bus:
  `artifacts/checkpoints/amodal_input_bus_complementary_seed145001.pt`
- Input-bus SHA-256:
  `4ae96f60b99107834c27840b8841e8b2ba20c10e6565c220b5607fb9c80d3c71`
- Training, replication, shuffled-reward, rejected-curriculum, and 4,096-case
  reports are stored beside this README.

## Honest boundary and next frontier

This closes synchronous complementary N=2 composition for one cognitive
primitive. It does not yet establish asynchronous streams, learned delay
handling, arbitrary N at behavioral scale, or replicated cross-renderer
transfer. Those remain Gate 4b.

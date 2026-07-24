# Success-weighted novelty design

Status: deferred until temporal rule binding passes representation and reversal-causality gates.

## Purpose

Reward successful behavior more strongly on task families that are less familiar to the learner.
The intended pressure is toward reusable operations rather than narrow solutions whose value ends
when one task family is mastered. Novelty alone is never rewarded: an unfamiliar failure receives
no curiosity bonus.

## Verifier-side novelty

Novelty is computed privately by the task generator from deterministic metadata such as primitive
family, identity count, cue family, sequence length, distractor structure, and composition depth.
None of this metadata is exposed to the agent. Avoid learned novelty estimators until an exact
generator-distance baseline has been exhausted; prediction-error novelty is gameable and can reward
irreducible noise.

## Objective

Use novelty as a multiplicative task-loss weight, not an additive reward:

`weighted_loss = normalized_task_loss * clipped_novelty_weight`

Normalize behavioral improvement relative to the primitive's chance level before comparing or
weighting tasks. Bound and batch-normalize novelty weights so a rare family cannot dominate an
update merely because of scale.

## Learning-speed reward

Reward sample-efficient adaptation separately from execution latency. For a lifetime, evaluate
held-out accuracy after a fixed sequence of support counts (for example zero, one, two, four, and
eight). Normalize every point above the primitive's chance level and compute an early-learning AUC.
A system that reaches the same final accuracy after one demonstration should score higher than one
that needs eight.

Use a lexicographic or gated objective:

1. correctness and final held-out accuracy are primary;
2. retention must remain above its declared floor;
3. only then add a bounded early-learning-AUC bonus;
4. add the already planned small response-latency bonus last.

Measure learning speed in unique experiences and optimizer updates, not wall-clock time, because
hardware and batching otherwise contaminate the cognitive signal. Report wall-clock throughput as
an engineering metric alongside it. Never reward training-set loss reduction directly: the speed
bonus must be computed on held-out behavior so rapid memorization or guessing cannot earn it.

## Safety gates

- Preserve the established spatial and shape retention floor on matched held-out evaluations.
- Report unweighted metrics alongside weighted training objectives.
- Audit generator metadata for accidental correlations with answers.
- Compare against uniform weighting and difficulty-only weighting.
- Reject any improvement that disappears under paired causal interventions.

## Activation criterion

Do not activate novelty weighting while the temporal storage blocker remains. First require:

1. the demonstrated rule is recoverable from raw writes and compact memory;
2. behavior changes correctly when visual sequence order is reversed;
3. spatial and shape retention remains within the declared floor.

Only then test whether success-weighted novelty improves compositional transfer to unfamiliar task
families.

## Training-data diversity and rendering invariance

The event-snapshot diagnostic establishes a useful training-diet rule. At a matched 4,096 examples,
4,096 distinct logical lifetimes reached 94.51% held-out while 1,024 logical lifetimes rendered four
ways reached 76.54%; 256 lifetimes rendered four ways remained at 54.27%. Distinct verifier-generated
tasks are therefore the primary source of compositional generalization. Render-seed augmentation is
layered on top to accelerate early learning and suppress nuisance memorization, never used as a
replacement for logical diversity. All variants of one logical lifetime must remain in the same
train/test partition.

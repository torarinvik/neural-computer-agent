# Temporal-to-spatial zero-label transfer

## Question

Does experience that formed a temporal first/last intention reduce the reward
bits required to learn a different primitive: infer whether a simultaneously
displayed selected object was on the left or right?

This is the first cross-primitive test. It follows the successful actuator
transfer but changes both the cognitive relation and the external protocol.

## Spatial primitive

Each lifetime contains:

1. one RGB frame showing two colored objects simultaneously;
2. one RGB feedback frame showing only the selected object's identity.

The private verifier asks whether the selected object occupied the left or
right position. Training and held-out evaluation use disjoint color-pair sets.
No task ID, position label, correct action, or unattempted-action target enters
the learner.

The output protocol maps the two possible intentions to two seed-specific
codes among four commands. The other two commands are distractors. Thus a
temporal two-action decoder cannot be reused directly.

## Compared initializations

- temporal intention, trainable on the spatial task;
- temporal intention frozen, fresh protocol adapter only;
- fresh intention and adapter over the experienced predictive core;
- fully fresh visual encoder, GRU, intention, and adapter;
- action-shuffled and reward-shuffled temporal-initialized controls.

All phase-B arms use the same logged spatial experiences, reward-bit prefixes,
200 optimizer updates, and 6,000 replayed examples per point.

## Metrics

- AULC above the 50% verified-accuracy majority floor;
- reward bits to 55%, 65%, and 75%;
- experienced/fresh reward-bit ratio;
- final accuracy only as a secondary metric.

## Causal audits

- **Horizontal mirror:** reverse the two visible object positions while keeping
  the selected feedback identity fixed. The correct left/right relation and
  protocol command must flip.
- **Missing feedback:** remove the selected-object frame. Accuracy must return
  toward the majority floor.
- **Opposite-rule stale state:** pair every query with a state from a lifetime
  carrying the opposite private relation.
- **Protocol swap:** exchange the two valid command codes without
  recalibration; performance must collapse predictably.
- **Action and reward shuffles:** neither control may cross a threshold.

All causal interventions are valid rerendered sensory streams. Hidden-state
swaps are used only for the explicitly labeled stale-state offline audit.

## Promotion gate

One seed advances only if the trainable temporal initialization:

- exceeds every control, including fresh, by at least 0.03 AULC;
- reaches at least 60% final accuracy;
- reaches a threshold in fewer reward bits than fresh;
- reaches at least 60% mirrored relabeled accuracy with at least 50% command
  flips;
- scores at most 55% with feedback removed;
- scores at most 40% under guaranteed opposite-rule stale states.

Three seeds are required for an exploratory cross-primitive milestone. Six
different primitives and three seeds per curriculum position remain required
before claiming compounding learning.

## Seed-211 result (bounded negative)

The single pre-registered seed did not pass, so no additional seeds were run:

- temporal intention, trainable: 78.39% final, 0.2427 AULC;
- temporal intention, frozen: 75.26% final, 0.2281 AULC;
- fresh intention: 79.43% final, 0.2396 AULC;
- trainable temporal advantage over fresh: +0.0031 AULC;
- every valid arm crossed 55/65/75% at the same 32/128/256 reward bits.

The learned spatial behavior itself was causal: horizontal mirroring produced
81.77% relabeled accuracy and 60.16% command flips; missing feedback returned
to 48.70%; opposite-rule stale state and protocol swapping each reduced
accuracy to 21.61%. Action/reward-shuffled controls crossed no threshold.

Thus temporal intention initialization did not measurably accelerate this
spatial primitive within the tested budget. The fresh-intention arm shared the
experienced predictive encoder and GRU, so the result rejects task-specific
intention-weight transfer, not possible reuse already present in the shared
predictive core. A fully fresh-core factorial cell is the cheapest next
localization.

## Fresh-core factorial localization

The missing fully fresh cell was then added with identical logged spatial
lifetimes, commands, rewards, head initialization, updates, and examples:

- predictively experienced core + temporal intention: 82.03% final,
  0.2458 AULC;
- predictively experienced core + fresh intention: 79.17% final,
  0.2172 AULC;
- fully fresh core + fresh intention: 50.00% final and 0.0000 AULC at every
  prefix through 510 reward bits.

This localizes a large transfer benefit to the shared visual/recurrent core,
but does not yet show that correctly paired prediction caused it. Generic
visual exposure, optimizer updates, or paired predictive structure are still
confounded.

The next and only promoted control is an identical core initialized from the
same weights and trained on the same rendered sequences, steps, and predictive
loss while future targets are shuffled across lifetimes. A paired-core
representation-transfer signal passes provisionally only if its fresh
intention:

- beats the shuffled-future core by at least 0.03 AULC;
- crosses at least one threshold with fewer reward bits;
- reaches at least 60% final accuracy;
- preserves the mirror, missing-feedback, and opposite-rule causal gates.

One seed remains provisional. If this gate fails, generic visual training—not
structured predictive reuse—is the supported explanation.

## Three-seed predictive-core result

The paired-versus-shuffled predictive-core gate passed on seeds 211, 257, and
313:

| Metric | Seed 211 | Seed 257 | Seed 313 | Mean |
|---|---:|---:|---:|---:|
| Paired core + fresh intention, final | 80.21% | 73.96% | 81.51% | 78.56% |
| Shuffled-future core, final | 70.57% | 46.35% | 57.81% | 58.25% |
| Fully fresh core, final | 50.00% | 50.00% | 50.00% | 50.00% |
| Paired core AULC | 0.2151 | 0.2125 | 0.2417 | 0.2231 |
| Shuffled-future core AULC | 0.1703 | 0.0083 | 0.0760 | 0.0849 |
| Paired AULC advantage | 0.0448 | 0.2042 | 0.1656 | 0.1382 |
| Mirror relabeled accuracy | 80.99% | 77.08% | 83.07% | 80.38% |
| Mirror prediction flips | 61.20% | 51.04% | 64.58% | 58.94% |
| Missing-feedback accuracy | 51.56% | 47.66% | 52.86% | 50.69% |
| Opposite-rule stale accuracy | 19.79% | 26.04% | 18.49% | 21.44% |

The paired predictive core reached 75% at 256 spatial reward bits on all three
seeds. Neither the shuffled-future core nor the fully fresh core reached 75% on
any seed by 510 bits.

Task-specific temporal intention reuse did not add a stable benefit. Its mean
AULC was 0.2234 versus 0.2231 for a fresh intention over the same paired core.
It helped early on seed 211, tied approximately on seed 257, and hurt the 75%
threshold on seed 313. The reusable asset is therefore the structured
predictive visual/recurrent core, not the old output head.

This is the first three-seed zero-label cross-primitive representation-transfer
milestone: correctly paired predictive experience on temporal streams made a
different simultaneous spatial relation learnable with fewer reward bits than
equal-compute shuffled pairing or no predictive experience. It is one
transition, not evidence of a compounding trend.

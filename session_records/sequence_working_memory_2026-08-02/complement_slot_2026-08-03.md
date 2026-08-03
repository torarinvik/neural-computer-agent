# Adjacent complement primitive — 2026-08-03

This is a small adjacent-primitive experiment, not a claim that span eleven is
mastered. The inherited span-nine/span-ten controller received one
zero-output successor slot. The new task used a third, visibly distinct
operation cue and required the complementary binary action (`1 - sequence`).
The learner received only controller-visible latent features, attempted opaque
actions, and scalar verifier outcomes; correct actions and the complement rule
were not stored in the replay buffer.

## Training arms

- Truthful arm: seed `93748`, 256 fresh logical lifetimes, 128 epochs, two
  distractors, binary outcome loss, hidden successor gate, and `.01` residual,
  gate, and inherited-logit replay penalties.
- Outcome-shuffled control: identical seed/budget and architecture, but the
  observed outcomes were permuted before training (seed `93749`).
- The first short truthful arms were non-flat but under-trained. The longer
  256-lifetime arm was retained because it crossed the data/phase-transition
  boundary without opening a larger old-skill regression.

## Independent audit

The audit used 1,024 lifetime-disjoint episodes with two distractors and a
fresh seed. It evaluated the parent, the trained child, the child with every
parameter in the appended slot zeroed, and the shuffled-outcome child.

| Check | Parent | Truthful child | Zeroed slot | Shuffled child |
| --- | ---: | ---: | ---: | ---: |
| Complement accuracy | 50.64% | **59.77%** | 50.64% | 47.56% |
| Operation-cue blank | 45.39% | **45.16%** | 45.39% | 45.61% |
| All memory reset | 50.01% | 50.00% | 50.01% | 50.00% |

The truthful child therefore has a **9.12-point causal gain** over its
zeroed-slot control. The old skills were preserved within the registered
two-point gate:

- span nine: 86.19% → 86.18% (`−0.01` point)
- span ten: 83.52% → 83.48% (`−0.04` point)

The cue-blank and full-reset controls return to chance, so the effect depends
on the public complement cue and the retained working-memory path. The
outcome-shuffled child remains at chance (47.56%) and does not reproduce the
truthful gain. The complete machine-readable audit is
`complement_slot_audit_seed294800.json`.

## Truthful replication

A second truthful seed (`93750`) used the same 256-lifetime/128-epoch recipe.
On a separate 1,024-lifetime audit it reached **58.12%**, while its zeroed-slot
control returned to **50.61%**, for a **7.51-point causal gain**. The matched
outcome-shuffled replica (`93751`) scored **47.47%**. Old-skill retention for
the truthful replica changed by −1.23 points on span nine and −0.86 points on
span ten; both remain inside the two-point gate. Its cue-blank and reset
controls were 47.36% and 49.93%.

The independent replication audit is
`complement_slot_audit_seed294950_with_shuffled.json`.

## Interpretation and next gate

This is now a replicated adjacent primitive that passes the five-point causal
promotion bar and the two-point old-skill retention gate in two truthful seeds.
It is still a **partial acquisition result** (58–60%, not mastery): persistent
experience and the inherited controller support useful transfer to a new,
visibly separable operation, but the primitive is not yet reliable enough for
deployment. The span-eleven successor branch remains paused; its input probe
already showed that its information is present, while outcome-to-slot credit
is still the bottleneck.

That protected continuation path was then implemented and smoke-tested; it
preserved the slot but did not improve it. The next data-scaling curve reached
61.5% at 512 fresh lifetimes and 66–69% at 1,024, but only some 1,024 seeds
passed old-skill retention. The complete curve and rejected arms are recorded
in `complement_data_scaling_2026-08-03.md`; do not scale span eleven until the
context-selective plasticity variance is reduced.

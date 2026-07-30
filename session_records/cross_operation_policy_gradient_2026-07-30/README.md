# Reward-grounded cross-operation transfer — 2026-07-30

## Question

Can an acquired numerosity comparison accelerate a different operation on the
same perceptual facts, using only pixels, the controller's sampled opaque
action, and that action's scalar verifier outcome?

The parent chooses the larger of two visible count fields. The adjacent task
uses a distinct public visual operation cue and asks for the inverse response:
choose the smaller field. No semantic task ID, count, correct action, or
unattempted-action label enters the learner.

## Selected procedure

- parent:
  `artifacts/checkpoints/unified_pair_numerosity_second_compounding_seed24031.pt`;
- target and numerosity replay blend: `0.248`;
- one newly appended zero-output adapter reads the frozen parent's accumulated
  amodal intention;
- exact uniform action logging;
- attempted-action policy-gradient loss with the task-agnostic binary chance
  baseline `outcome - 0.5`;
- cosine learning rate `0.003 → 0.0003`, beginning after 25% of updates;
- 128 optimizer updates × 8 new lifetimes;
- 1,024 unique new lifetimes / 6,144 verifier bits;
- balanced replay of relation, magnitude, and inherited numerosity.

The policy loss changes only the probability of the action actually attempted.
Both success and failure carry signal, but the unattempted answer is never
constructed or used as a target.

## Replicated result

All scores use 8,192 fresh held-out lifetimes.

| seed | truthful outcomes | outcomes shuffled | inherited intention ablated |
|---:|---:|---:|---:|
| 25021 | 57.42% | 38.48% | 38.53% |
| 25031 | 56.98% | 38.21% | 36.29% |
| **mean** | **57.20%** | **38.34%** | **37.41%** |

The truthful arm therefore gains:

- **+18.86 points** over the matched shuffled-outcome control;
- **+19.79 points** over the matched zero-content intention control.

The two truthful replicas retain the three inherited skills at:

- pair relation: 95.31–95.87%;
- visible magnitude: 82.53–83.39%;
- inherited numerosity: 84.81–85.72%.

Each complete training/evaluation run took 12–16 seconds locally on MPS. The
deployed controller still uses one pass per event.

## Why the learning rule matters

The earlier independent-logit BCE rule learned the target, but shuffled
outcomes could drift the binary decision toward chance and reached 51–53% at
the same 128-update budget. It therefore left only a small truthful-versus-
shuffled gap.

The selected attempted-action policy gradient preserves truthful learning while
both shuffled controls remain below 39%. It is a cleaner use of the same
learner-visible information, not additional supervision.

## Rejected tiny forks

- Direct equality remained near chance even after 2,048 lifetimes. Disposable
  probes found the relation poorly represented, so this was a curriculum leap,
  not a reason to scale training.
- A feature-wise multiplicative intention operator did not beat the additive
  adapter.
- A lower-capacity scalar intention operator ignited earlier on one seed but
  did not beat the additive adapter on a fresh matched seed.
- Forcing two matched trajectories to attempt opposite actions doubled
  verifier bits and did not improve learning.

Those mechanisms were removed from the promoted implementation.

## Claim boundary and next frontier

This is verified **cross-operation transfer**, not mastery. The controller now
uses an acquired comparison intention and truthful scalar outcomes to learn an
inverse operation substantially better than either causal control, at half the
2,048-lifetime budget of the first replication.

No stable mastery threshold has yet been reached, so stable
bits-to-threshold and a finite mastery transfer ratio are intentionally
unreported. The next frontier is a gradual, verifier-only operation curriculum
that raises held-out performance from 57% to reliable mastery while reducing
new lifetimes further.

The six machine-readable reports in `reports/` are the authoritative evidence.

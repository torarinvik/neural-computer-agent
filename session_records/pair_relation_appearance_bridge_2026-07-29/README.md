# Cross-appearance relation bridge — 2026-07-29

## Breakthrough

The unified controller now holds one same/different relation across both its
original bar objects and a second, independently rendered diamond contour
family while retaining binary mapping, visible context, and visible-context
XOR. The learner used only rendered RGB, opaque attempted actions, scalar
verifier outcomes, its own latent state, and opaque behavior rehearsal.

The promoted seed-9303 checkpoint passed on 4,096 held-out lifetimes:

| capability | accuracy | complete causal gate |
|---|---:|:---:|
| original bars relation | 99.61% | pass |
| new diamonds relation | 97.52% | pass |
| binary hidden mapping | 93.37% | pass |
| visible context | 91.74% | pass |
| visible-context XOR | 90.58% | pass |

The independent audit reproduced 99.61% bars and 97.74% diamonds. Blank vision
and missing-second-object controls were at 49–50%, and valid pixel
counterfactuals passed their accuracy and prediction-flip gates. Every
non-relation parent parameter remained bit-identical.

The curated checkpoint is
`artifacts/checkpoints/unified_pair_relation_appearance_bridge_seed9303.pt`,
SHA-256
`3fbb53049a1ecb5496c308eba195371531d1ec87c8be3edb0c3ddf980a0b9919`.

## Causal reuse control

The matched reset arm used the same controller architecture, 64-unit nonlinear
gate refiner, pixels, seeds, optimizer, 320 updates, rehearsal, and 184,320
total verifier bits. The only intervention was replacing the learned relation
slot with its original zero-output initialization.

It learned diamonds to 97.69% and retained the three unrelated tasks, but bars
fell to 72.03% and failed every bars counterfactual/mastery gate. The
experienced arm retained bars at 99.61%. Thus capacity and diamond experience
alone do not explain the combined capability: the inherited bars relation is
causally necessary.

## How the result was localized

The first direct bridge looked like severe negative transfer: the experienced
slot stayed near 27% diamonds while the reset slot learned. Several cheap
controls separated four causes:

1. a bars-to-diamonds pixel morph and a mixture curriculum did not solve it;
2. a temporary ReLU gate leak did not solve it, rejecting the dead-gate
   hypothesis;
3. removing protection let the experienced slot reach 99.90% diamonds,
   preserve bars at 99.64%, and transfer zero-shot to dot pairs at 95.05%, but
   it disturbed unrelated skills;
4. a locality penalty large enough to protect old tasks suppressed acquisition.

The missing operation was consolidation. A new 64-unit nonlinear gate refiner
is exactly zero at insertion, so the expanded model is bit-identical before
training. Training was split:

- **acquisition:** broaden the existing relation residual on diamond outcomes;
- **consolidation:** freeze that residual and the old linear gate, snapshot the
  verified successful controller, and train only the refiner by opaque
  self-distillation plus old-behavior rehearsal.

This separates learning *what relation to compute* from learning *where that
computation may speak*.

## Accounting

- acquisition: 32 updates;
- consolidation: 288 updates;
- new relation lifetimes: 10,240;
- new relation verifier bits: 61,440;
- four replay streams: 5,120 lifetimes each;
- total verifier bits including rehearsal: 184,320;
- measured end-to-end time: 24.90 seconds.

The reset control used exactly the same accounting.

## Honest boundary

This is a verified capability and causal reuse result, not yet a robust fixed
sample threshold. Seed 9301 passed at 256 consolidation updates; seed 9303
passed at 288. Seed 9302 was 0.034 percentage points below the XOR gate at 256
and over-consolidated diamonds by 288. Adaptive stopping or population
selection must therefore replace a fixed duration before this becomes a robust
sample-efficiency recipe.

Disconnected dot pairs remain an unmastered third contour at 70.61% on the
promoted audit. The next gradual repertoire rung should use the same
acquire-then-localize mechanism for dot pairs and test whether two retained
contours make the third cheaper than the matched reset lineage.

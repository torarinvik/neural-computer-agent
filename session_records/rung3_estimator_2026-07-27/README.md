# Re-measuring the third rung: why its estimator was replaced

This sweep exists because the third rung's committed three-seed table did not
reproduce. Rerunning the identical configurations on CUDA moved the stable
thresholds by up to twelve updates. Runs on that device are bit-deterministic —
verified by repeating one configuration and comparing every evaluation field —
so nothing but floating-point rounding differed between the two devices.

An estimator that swings twelve updates on rounding alone cannot support a claim
about an eight-update saving.

## What was run

208 runs: `train_compounding_transfer` on seeds 8411–8418, budgets 40 to 88 in
steps of four, both arms, 512 held-out lifetimes per evaluation.

- baseline arm: `unified_memory_online_utility_seed6810.pt` (binary mapping only)
- experienced arm: `unified_binary_context_integrated_seed8397.pt`
  (binary mapping plus direct context)

## The committed estimator, on eight seeds

Stable threshold means the first budget whose gates pass and whose every later
measured budget also passes.

| Seed | Baseline | Experienced | Ratio |
|---|---:|---:|---:|
| 8411 | 68 | 60 | 1.133 |
| 8412 | 76 | 56 | 1.357 |
| 8413 | 60 | 60 | 1.000 |
| 8414 | 88 | 60 | 1.467 |
| 8415 | 80 | 56 | 1.429 |
| 8416 | 64 | 68 | 0.941 |
| 8417 | 68 | 64 | 1.063 |
| 8418 | 60 | 80 | 0.750 |

Median 1.098, range 0.750 to 1.467, with the **baseline faster on two seeds**.
The committed run's exactly-eight-updates-on-three-of-three was luck. The metric
is a step function over single runs: seed 8412's baseline passes at 64, fails at
68 and 72, then passes from 76, and the definition therefore reports 76.

## The replacement

Interpolate each seed's held-out accuracy curve to a fixed target and pair
within seed. This is monotone in the underlying quantity and independent of
where a pass/fail boundary happens to land.

| Target | Median ratio | Mean | Seeds above 1 | Sign test |
|---|---:|---:|---:|---:|
| 85% | **1.231** | 1.225 | 8 of 8 | p = 0.0078 |
| 90% | 1.192 | 1.174 | 7 of 8 | p = 0.0703 |

Backed by the paired accuracy difference at every budget: the experienced arm is
ahead at all thirteen, by +13.4 points at the smallest budget narrowing to
+0.5 as both saturate. Pooled over 104 paired cells, +4.81 points with 80 wins
against 23 losses, sign test p = 1.5e-8.

**The finding is real and larger than was reported. Only the estimator failed.**

The fourth rung and the growth comparison are in
`../rung4_race_2026-07-27/`.

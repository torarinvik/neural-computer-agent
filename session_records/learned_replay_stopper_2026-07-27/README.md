# Cross-generation learned replay stopping

## Question

Can a task-agnostic controller trained on earlier five-action learning
trajectories decide how much replay a new six-action learner deserves, while
preserving verifier/sample efficiency and reducing optimizer work?

The representation diagnostic passed strongly, but the behavioral stopping
policy did not replicate within the pre-registered sample-efficiency gate. No
checkpoint is promoted.

## Diagnostic localization

Fixed replay traces exposed only generic learning state available at decision
time: full observed-experience loss, previous loss reduction, previous gradient
norm, observed-example count, and replay position. The target was generated
from subsequent verified loss reduction; it used no task identity, action
label, correct answer, or hidden environment state.

A one-update predictor trained on two five-action streams transferred to two
held-out six-action streams:

- held-out correlation: `0.79–0.83`;
- sign accuracy: `74–76%`;
- MAE improvement over the mean predictor: `18.6–28.5%`;
- shuffled-target correlation: approximately zero.

Despite passing the diagnostic gate, the one-update policy stopped inside
multi-update learning valleys and lost behavioral mastery. This falsified
one-step loss reduction as a sufficient stopping objective.

An eight-update target supplied task-agnostic patience. With four unique
five-action training streams, two independently initialized probes achieved:

| Probe | Held-out MAE improvement | Correlation | Sign accuracy | Shuffled correlation |
|---|---:|---:|---:|---:|
| seed 8125 | `23.25%` | `0.9140` | `62.50%` | `0.0471` |
| seed 8126 | `24.40%` | `0.9173` | `65.62%` | `0.0502` |

Both passed the frozen representation gate.

## Behavioral smell test

On matched seed 8120:

| Policy | Stable verifier bits | Updates | Final utility | Result |
|---|---:|---:|---:|---|
| fixed replay 16 | `4,560` | `1,024` | `0.87888` | baseline |
| eight-step predictor, zero compute price | `4,800` | `920` | `0.87939` | promising |
| eight-step predictor, `0.0005` price | `7,440` | `800` | `0.86844` | rejected |

The zero-price policy saved `10.16%` of updates with a `5.26%` verifier-bit
delay and therefore advanced to prospective testing. The priced arm bought
more compute savings by materially damaging sample efficiency.

## Prospective matched replications

The frozen gate required:

- the behavioral capability gate to pass;
- at least `10%` fewer composition updates;
- no more than `10%` regression in stable verifier bits.

### Single predictor

| Seed | Learned/fixed bits | Update saving | Utility delta | Capability |
|---:|---:|---:|---:|---|
| 8127 | `8,160 / 8,400` | `14.45%` | `+0.00037` | pass |
| 8128 | `9,120 / 6,480` | `20.31%` | `-0.00543` | fail |

The second seed stopped overconfidently and was rejected.

### Conservative unanimous ensemble

| Seed | Learned/fixed bits | Update saving | Capability |
|---:|---:|---:|---|
| 8127 | `9,360 / 8,400` | `1.95%` | pass |
| 8128 | `6,960 / 6,480` | `9.77%` | pass |

Unanimity restored capability but failed the compute-saving gate.

### Ensemble mean

| Seed | Learned/fixed bits | Update saving | Utility delta | Capability |
|---:|---:|---:|---:|---|
| 8127 | `8,160 / 8,400` | `12.89%` | `+0.00298` | pass |
| 8128 | `7,200 / 6,480` | `17.97%` | `-0.00486` | pass |

This was the best fork, but seed 8128 required `11.11%` more verifier bits,
missing the frozen `10%` limit. It is not promoted.

## Conclusion

The experiment establishes a useful but bounded result:

> Marginal replay value is strongly predictable across task generations, but
> local replay-loss reduction is not yet a robust proxy for behavioral
> bits-to-mastery.

The next high-ROI target is a longer-horizon predictor trained against verified
behavioral learning progress and retention, with computation explicitly
charged. It should predict whether a block of processing improves future
held-out verifier utility—not merely whether it lowers replay loss.

All raw trace data and temporary predictor weights remain disposable cloud
artifacts. The compact reports here preserve every promoted/rejected decision.

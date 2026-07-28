# Full-controller temporal memory volatility

## Breakthrough

The unified visual controller learned to protect or rewrite individual
disk-backed latent memories from temporal verifier evidence. The result uses no
task IDs, semantic row labels, correct replacement labels, or replay.

The parent already used seven generic replacement features. Volatility was
added as an eighth feature through the existing zero-initialized residual. Only
one new scalar coefficient trained; inherited columns were restored exactly
after every optimizer step.

## Confound-resistant task

Each bank contains three stable and three decoy latent skills plus one incoming
skill. Every stored row has:

- the same ten accesses;
- five verified successes;
- five verified failures;
- no semantic category visible to the controller.

Stable rows receive failures then successes. Decoys receive successes then
failures. Their aggregate reliability is identical; only verifier **order**
changes volatility. Future pixel-rendered queries cover the three stable skills
and the incoming skill, so replacing a decoy is the only way to preserve all
verified capabilities.

## Learning result

Both independent normal runs used 32 updates, 64 banks per update, and no
replay. Stable mastery is the first measured prefix at or above 95% valid
replacement whose every later prefix also remains above 95%.

| seed | preflight valid | stable update | bits to stable | held-out valid | held-out accuracy | oracle |
|---:|---:|---:|---:|---:|---:|---:|
| 17105 | 48.44% | **24** | **6,144** | **99.61%** | 96.58% | 96.39% |
| 17107 | 44.14% | **24** | **6,144** | **98.83%** | 97.07% | 97.36% |
| reward-shuffled 17106 | — | never | never | 57.81% | 89.84% | 96.00% |

Each normal training loop took about 45 seconds; complete training plus all
held-out and retention audits took 72–80 seconds, safely below the five-minute
cap. Each full run consumed 8,192 verifier bits and 28,672 unique logical
contexts.

## Causality and retention

- Shuffling volatility among rows cut valid replacement to 47–49%.
- Reversing only outcome order made the learned policy evict a stable row on
  98.4–99.6% of banks.
- Reward shuffling prevented stable learning.
- The old reliability-dominant memory utility passed.
- Binary mapping and four-rule behavioral gates passed.
- Only `memory_replacement_extra_gate.weight` changed; its inherited columns
  remained bit-identical.

## Physical disk audits

Three 128-bank audits used real `DiskLatentMemory` rows, ordinary
content-addressed reads, verified outcome updates, elastic replacement, and
save/reload before and after replacement.

| audit | valid replacement | visual accuracy | shuffled valid | constant valid | reversed stable eviction |
|---|---:|---:|---:|---:|---:|
| 17201 | **100%** | 94.34% | 50.00% | 46.88% | **100%** |
| 17202 | **100%** | 95.12% | 52.34% | 49.22% | **100%** |
| selected 17203 | **100%** | 91.80% | 52.34% | 51.56% | **100%** |

All 384 banks preserved keys, values, usage, access, success, failure, and
volatility tensors exactly through disk reload. Every bank stayed at capacity
six.

## Important boundary discovered

The first physical audit used the naturally varying learned admission strengths
stored in the parent. Some exact content queries were redirected to a different
row because retrieval adds `log(write_strength)` as a prior. Outcome histories
therefore accrued on the actually retrieved rows rather than the presumed
source rows, and valid replacement reached only 67.19%.

The promoted physical isolation holds admission strength equal while histories
are learned. This is not hidden: it identifies the next architectural frontier.
The system needs an explicit causal receipt identifying which row produced a
read and outcome, and likely needs to separate retrieval confidence from
admission strength.

## Promoted checkpoint

`artifacts/checkpoints/unified_controller_memory_volatility_seed17107.pt`

SHA-256:
`da89893ffe67a20907755c48b4dfbd0755a1469dd05ed7211473e07d95c21c07`

The parent is unchanged except for expansion from seven to eight replacement
features and the learned volatility coefficient. The selected state is
reconstructed exactly from the sole changed scalar recorded in the seed-17107
report.

## Next rung

Return the exact retrieved row index as a generic causal receipt, attribute each
verifier outcome to that receipt, and test unequal admission strengths. The
acceptance gate is:

1. at least 95% valid replacement under unequal priors;
2. at least a 30-point loss when receipts or volatility are shuffled;
3. exact disk persistence and bounded capacity;
4. old utility and behavioral retention;
5. no run longer than five minutes.

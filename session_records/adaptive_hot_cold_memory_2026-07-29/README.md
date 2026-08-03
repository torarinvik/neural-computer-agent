# Verified adaptive hot/cold latent memory

## Breakthrough

The neural computer now has a lossless two-tier external memory:

- **cold disk memory** preserves every consolidated latent representative;
- **hot RAM/VRAM memory** contains the small working set needed now;
- a scalar causal-rescue trace promotes useful cold diversity into the hot set;
- unused reserve automatically thaws out of hot memory without being deleted
  from disk;
- if the difficult distribution returns, verified failures trigger sparse cold
  retries and the useful reserve is reactivated.

The mechanism sees no appearance name, rule bit, semantic row label, or correct
memory action. A rescue is one generic event: the current hot attempt failed,
a cold deep read succeeded, and that successful read used a reserve row.

The selected update is:

`protection = min(1, 0.90 × protection + causal_rescue)`

The reserve is hot while protection is at least `0.5`. These generic
hyperparameters were selected by a separate verifier-side population race, not
chosen from the graduation seeds.

## Nonstationary protocol

Every run contains 4,096 independent six-row cold archives:

1. 16 rounds of disconnected dot-pair queries;
2. 24 rounds of easy bar queries, where the reserve is unnecessary;
3. 12 rounds of dot-pair queries returning.

The schedule is private generator metadata. The memory policy observes only
controller-created latents, its attempted result, scalar verifier success, and
occasional cold-read results.

## Three untouched graduation replicas

| metric | seed 52001 | seed 53001 | seed 54001 |
|---|---:|---:|---:|
| initial hard: adaptive hot | 98.7587% | 98.7488% | 98.7885% |
| initial hard: fixed core | 98.6664% | 98.6359% | 98.6893% |
| easy-phase adaptive hot rows | **4.083** | **4.081** | **4.069** |
| easy-phase cumulative rows | 4.261 | 4.263 | 4.243 |
| full cold rows | 5.994 | 5.996 | 5.995 |
| easy-phase adaptive accuracy | 99.9680% | 99.9781% | 99.9985% |
| returned hard: adaptive hot | **98.7213%** | **98.7162%** | **98.6918%** |
| returned hard: fixed core | 98.6359% | 98.6064% | 98.6074% |
| returned hard: shuffled evidence | 98.6471% | 98.6237% | 98.6216% |
| returned hard after sparse cold retry | **99.1567%** | **99.2188%** | **99.1394%** |
| last-four gain over fixed core | 0.1160 pp | 0.1404 pp | 0.1160 pp |
| last-four gain over shuffled evidence | 0.0946 pp | 0.1160 pp | 0.1038 pp |
| corrupted cold archive | 49.85% | 50.10% | 50.37% |
| hot/cold serialized-byte ratio | 70.37% | 70.47% | 69.93% |

All thirteen substantive graduation gates pass on every seed.

The returning-task curve is particularly important. In the first four rounds,
adaptive hot memory is only 0.027–0.052 percentage points above fixed core. By
the final four rounds, the paired advantage grows to 0.116–0.140 points.
Shuffling the same rescue evidence across compatible banks removes most of the
gain. This is online reactivation from experience, not a static lucky subset.

Only 2,877–2,947 cold retries are policy-visible among 425,984 hot events
(0.68–0.69%). The vectorized audit computes all cold counterfactuals to measure
the ceiling, but non-retry outcomes are not delivered to the policy.

## Decay population race

Six rates shared the same 2,048-bank search stream:

| decay | accepted | easy hot rows | returned accuracy | late gain vs fixed |
|---:|:---:|---:|---:|---:|
| 0.50 | no | 4.064 | 98.5881% | 0.0549 pp |
| 0.70 | no | 4.065 | 98.5921% | 0.0549 pp |
| 0.80 | no | 4.069 | 98.6125% | 0.0793 pp |
| **0.90** | **yes, selected** | **4.081** | 98.6450% | 0.1221 pp |
| 0.95 | yes | 4.114 | 98.6654% | 0.1770 pp |
| 0.97 | no | 4.185 | 98.6735% | 0.1770 pp |

Fast decay fails accuracy/reactivation gates. Slow decay `0.97` fails to thaw
enough stale context. Both `0.90` and `0.95` pass; the pre-registered
lexicographic selector first constrains accuracy and causality, then chooses
`0.90` for the smaller hot footprint.

Search compute is reported separately in `decay_selection.json`: 1,277,952 hot
events, 1,277,952 cold counterfactuals, and 2,555,904 fixed-baseline events.

## Physical implementation and adversarial audits

`TieredLatentMemory` is a real reusable memory object, not an audit-only mask.
Its archive stores:

- all cold keys, values, write strengths and histories;
- learned representative ranks;
- the generic protection scalar and activation threshold.

Across 384 audited archives:

- every cold tensor reloaded exactly;
- every hot tensor reloaded exactly;
- all tier metadata reloaded exactly;
- every archive promoted after rescue and thawed after seven quiet intervals;
- cold row count never changed;
- controller weights remained bit-identical.

Cold-value corruption reduces behavior to chance. Shuffling protection evidence
while preserving its per-round count and reserve-size stratum consistently
hurts reactivation; resulting mean hot footprints stay within 0.003 rows.
Fixed-core memory never develops the growing late-phase gain.

## Claim boundary

Demonstrated:

- one lossless disk archive and a smaller adaptive RAM/VRAM working set;
- verified promotion, thaw and later reactivation;
- a growing learning curve after the task returns;
- task-agnostic evidence placement is causally important;
- roughly 32% fewer logical hot rows than the full archive;
- exact physical persistence and no controller-weight forgetting.

Not yet demonstrated:

- transfer of the tiering policy across unrelated cognitive primitives;
- learned scheduling of *when* to probe cold memory;
- optimal hot capacity beyond the current two-level reserve;
- wall-clock benefits with genuinely large disk archives;
- long-horizon recurrence of many skills competing for the same hot budget.

## Artifacts

- `graduation_seed52001.json`
- `graduation_seed53001.json`
- `graduation_seed54001.json`
- `decay_selection.json`
- `race/decay_*.json`

Executable sources:

- `experiments/archive/unified_cognitive_controller/audit_adaptive_hot_memory.py`
- `experiments/archive/unified_cognitive_controller/select_hot_memory_decay.py`
- `experiments/archive/unified_cognitive_controller/memory.py`

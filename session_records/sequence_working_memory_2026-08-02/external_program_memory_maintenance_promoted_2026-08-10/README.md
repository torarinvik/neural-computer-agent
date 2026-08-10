# Promoted: learned maintenance over executable external memory

Date: 2026-08-10

Seeds: `25001`, `25002`, `25003`

Schema: `neural-computer.external-program-memory-maintenance.v1`

## Result

The shared register interpreter and canonical runtime are frozen. A generic
external maintenance policy receives only normalized file-storage telemetry
and a structural action mask. It learns from one scalar verifier utility per
fresh phase and chooses among `grow`, `share`, `compress`, `evict`, and
`defer`. Every selected mutation is executed through the real executable-file
transaction API: stable-prefix admission, held-out equivalence, retention
verification, or checksummed durable compression.

| seed | trained eval | fresh eval | shuffled-verifier eval | real actions observed | replay | controller updates |
| ---: | ---: | ---: | ---: | :---: | ---: | ---: |
| 25001 | 1.0000 | 0.6000 | 0.6000 | all 4 mutations | 0 | 0 |
| 25002 | 1.0000 | 0.2000 | 0.2000 | all 4 mutations | 0 | 0 |
| 25003 | 1.0000 | 0.4000 | 0.6000 | all 4 mutations | 0 | 0 |

All three runs pass real-transaction receipts, exact file persistence,
corrupt-compressed-payload rejection, canonical runtime traversal, frozen
interpreter, zero replay, and zero controller optimizer updates. The policy
reaches its result with `96` online policy updates and `96` unique verifier
bits per seed; raw verifier rows are not retained.

## Architectural meaning

This closes the seam between a learned generic maintenance decision and the
executable file backend. The controller remains a frozen compute substrate;
memory-side policy state can decide when to grow, share, compress, evict, or
defer, while the verifier remains the only authority allowed to commit a
change. Logical file IDs, candidate artifacts, equivalence probes, and
retention probes remain external to the controller and intention bus.

## Claim boundary

This promotes replay-free learned maintenance choice over a bounded synthetic
executable-file workload. It does not establish learned compression,
autonomous verifier design, unrestricted memory growth, arbitrary program
synthesis, genuine Brain Workshop program acquisition, or general continual
learning. The next decisive pressure test is to connect this policy to the
persistent multi-step frontier while acquiring executable capabilities from a
non-synthetic Brain Workshop family stream, with retention and storage costs
charged to the same utility.

Reports: `report_seed25001.json`, `report_seed25002.json`,
`report_seed25003.json`.

Reproduce with:

```bash
PYTHONPATH=src:. .venv/bin/python \
  experiments/external_program_memory_maintenance/train.py \
  --seed 25001 \
  --report-out /tmp/external-program-memory-maintenance-25001.json
```

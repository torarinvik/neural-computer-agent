# Promoted: verifier-gated executable-memory lifecycle

Date: 2026-08-10

Seeds: `24001`, `24002`, `24003`

Schema: `neural-computer.external-program-memory-lifecycle.v1`

## Result

The shared register interpreter and amodal controller are frozen. The
external file bank begins with a protected executable file, an equivalent
duplicate, and a distinct file. The memory-side lifecycle then rejects
protected eviction, rejects non-equivalent consolidation, evicts an
unprotected file after held-out retention verification, consolidates the
equivalent duplicate while retaining the survivor's logical ID, and commits
float16 durable storage only after decompression and behavior verification.

| seed | held-out verifier checks | files before | files after | storage after compression | retention |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 24001 | 448 | 3 | 1 | 14,016 / 28,032 bytes | 1.0000 |
| 24002 | 448 | 3 | 1 | 14,016 / 28,032 bytes | 1.0000 |
| 24003 | 448 | 3 | 1 | 14,016 / 28,032 bytes | 1.0000 |

Every run passes protected-eviction no-op, held-out retention, wrong-function
consolidation rejection, equivalent consolidation, stable logical identity,
corrupt-payload rejection, mutating-probe rejection, durable compression,
exact persistence, canonical runtime traversal, frozen executor, zero replay,
and zero controller-update gates.

## Claim boundary

This promotes a bounded, verifier-gated lifecycle contract for opaque
executable external memory. It does not establish learned maintenance-policy
selection, unrestricted memory growth, arbitrary new computation, or general
continual learning. The next pressure test is to place the lifecycle behind a
learned anonymous maintenance policy and run it over a longer nonstationary
Brain Workshop stream with retention and compression costs included.

Reports: `report_seed24001.json`, `report_seed24002.json`,
`report_seed24003.json`.

Reproduce with:

```bash
PYTHONPATH=src:. .venv/bin/python \
  experiments/external_program_memory_lifecycle/train.py \
  --seed 24001 \
  --report-out /tmp/external-program-memory-lifecycle-24001.json
```

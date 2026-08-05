# Breakthrough: opaque routing of independently acquired working-memory procedures

This record combines two independently acquired growth artifacts in the
canonical external artifact memory. Both artifacts use the same generic
successor-slot schema and the same frozen span-eight parent:

- `complement`: emit the bitwise complement of the retained sequence;
- `complement_reverse`: reverse the retained sequence, then complement it.

The parent was weak on both span-ten procedures. Each artifact was acquired
with the controller frozen, parent-logit rehearsal on spans 4/6/8, and scalar
outcomes from rendered episodes. A replaceable memory-side router then saw
only opaque query tensors, random opaque row keys, attempted rows, and scalar
outcomes. Procedure names and correct rows remained verifier-private.

## Acquisition

| Procedure | Seed | Parent | Child | Zeroed growth | Gain | Unique lifetimes | Unique verifier bits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| complement | 65005 | 29.7% | 76.4% | 29.7% | +46.7 pp | 8,192 | 57,344 |
| complement-reverse | 65006 | 27.8% | 63.8% | 27.8% | +35.9 pp | 8,192 | 57,344 |

Both acquisitions passed initial behavior preservation, frozen-core identity,
causal growth, retention within two points, exact rehydration, and corruption
rejection.

## Opaque routing and execution

| Router seed | Procedure | Selected | Wrong artifact | Zeroed artifact |
| ---: | --- | ---: | ---: | ---: |
| 66002 | complement | 76.2% | 30.0% | 29.2% |
| 66002 | complement-reverse | 63.6% | 26.9% | 26.9% |
| 66003 | complement | 75.3% | 28.6% | 28.8% |
| 66003 | complement-reverse | 63.9% | 27.2% | 27.2% |

Both router runs reached 100% held-out routing, 50% under reward-shuffled
outcomes, 100% under candidate-row permutation, and 50% for raw cosine-key
matching. Both wrong-address controls were behaviorally discriminative for
both procedures. Bank reload, frozen-core identity, and corruption rejection
passed in both runs.

This is a narrow breakthrough for the intended CPU/filesystem analogy:
external memory can hold multiple independently acquired executable states,
and a generic frozen controller can use learned opaque addressing to retrieve
the state required by the current working-memory computation. It is not yet a
claim of arbitrary program induction, general address discovery, or broad
intelligence. The procedure artifacts are acquired rather than cold-start
invented, and the router currently replays a small address-training set.

Reports:

- `../frozen_growth_span10_complement_512_2026-08-04/report.json`
- `../frozen_growth_span10_complement_reverse_512_2026-08-04/report.json`
- `report.json` — router seed 66002
- `../opaque_route_acquired_complement_bank_512_replication_2026-08-04/report.json` — router seed 66003

Harnesses:

- `experiments/working_memory_continuous/acquire_frozen_growth.py`
- `experiments/working_memory_continuous/route_acquired_procedure_bank.py`

# Promoted: outcome-only executable-file admission

Date: 2026-08-10

Seeds: `23001`, `23002`, `23003`

Schema: `neural-computer.external-program-file-admission.v1`

## Result

The shared register interpreter and amodal controller remain frozen. Two
opaque executable files are mastered first. A third candidate is staged,
tested through deterministic scalar verifier outcomes, and committed only
after a stable suffix of at least `32` outcomes clears the admission
threshold. A corrupted candidate is rejected without changing the file bank.

The warm learner adds a separate opaque external route cell for the new target
context. This is intentional: an earlier single global route experiment
showed that appending a new route into one shared policy can interfere with an
already mastered context. Separate cells preserve the old route while keeping
the controller frozen and the memory externally growable.

| seed | warm target | matched fresh | source retention | shuffled control | wrong-file control |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 23001 | 0.7917 | 0.8250 | 0.9167 | 0.1417 | 0.1708 |
| 23002 | 0.8208 | 0.8083 | 0.9292 | 0.1042 | 0.1583 |
| 23003 | 0.8750 | 0.8708 | 0.9292 | 0.0708 | 0.3458 |

All three runs pass the full promotion gate set: source mastery, stable
candidate admission, rejected-candidate no-op, protected source and new
files, correct source/target cell selection, warm and fresh target mastery,
complete-prefix source retention, shuffled-outcome and wrong-file controls,
exact file/cell-bank persistence, canonical runtime traversal, frozen
controller, zero replay, and zero controller optimizer updates.

The verifier uses a frozen reference bank that is private to the diagnostic
verifier. The learner receives only opaque event tensors, sampled choices,
exact propensities, and scalar outcomes. No program identity, relation,
target action, or verifier target is exposed to the learner.

## Accounting

Per seed: `2,100` unique logical lifetimes, `2,266--2,308` unique verifier
bits, `204--208` program-file verifier outcomes, `10,600` external route
decision updates, `5,300` external feedback updates, `500` interpreter
pretraining optimizer updates, `0` controller optimizer updates, `0` replayed
examples, and `0` retained raw verifier rows. The warm route reaches its
stable threshold in `600--800` target episodes; the matched fresh route uses
`600--1,000` episodes.

## Claim boundary

This promotes bounded verifier-gated admission of one portable executable
external file plus context-separated route-cell retention. It does not prove
program synthesis, arbitrary new computation, unrestricted memory growth, or
general continual learning. The next decisive pressure is candidate
generation from outcomes and a larger genuinely non-synthetic Brain Workshop
family stream, with the same complete-prefix retention and fresh controls.

Reports: `report_seed23001.json`, `report_seed23002.json`,
`report_seed23003.json`.

Reproduce with:

```bash
PYTHONPATH=src:. .venv/bin/python \
  experiments/external_program_file_admission/train.py \
  --seed 23001 \
  --report-out /tmp/external-program-file-admission-23001.json
```

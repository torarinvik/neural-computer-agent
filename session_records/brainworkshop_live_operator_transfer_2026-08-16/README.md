# Live operator staging across Workshop and two rendered mazes (2026-08-16)

Status: **development diagnostic; no holdout, promotion, or curated-bank
admission**.

This is the first audit that puts the verifier-gated operator boundary through
the real public loop:

```text
rendered Workshop frame -> learned event tensors -> one controller
-> rendered source maze -> rendered target maze -> rendered Workshop frame
```

The same `CanonicalBrainWorkshopAgent` instance was used for all four stages.
The maze wrappers received only learned event tensors, opaque feedback, and a
task-local world model.  Source-maze successor facts were not copied into the
target maze.  The operator candidate was staged only from source-maze scalar
returns, with a stable-prefix gate and a digest retention probe.

## Development result

| check | result |
| --- | --- |
| one core across Workshop/source/target/Workshop | **yes** |
| controller parameters unchanged | **yes** |
| source admission evidence | 4 eligible checkpoints |
| stable-prefix admission | **rejected** (`insufficient-stable-evidence`) |
| target operator use | **none** (fail-closed) |
| Workshop after maze | 4 verifier bits, 0.25 accuracy |

The source curve was `0.000, 0.000, 0.724, 0.545`; it never met the 0.70
stable-prefix requirement.  The target maze therefore received no unverified
operator.  This proves the live boundary and safe refusal, not positive
Workshop-to-maze operator transfer.  The gate must not be relaxed to make this
run pass.

## Next step

Keep this as a negative live control.  Improve the rendered maze evidence and
run a matched admitted-versus-fresh development battery only after a source
operator earns a stable prefix under ordinary controls.  Do not spend reserved
holdout seeds or admit anything to the curated bank from this record.

## Reproduction

```bash
PYTHONPATH=src .venv/bin/python -m experiments.brainworkshop_canonical.live_operator_transfer \
  --neural-workshop '/Users/torarinvikbjarko/Documents/Coding Projects/Python Projects/neural-workshop' \
  --output session_records/brainworkshop_live_operator_transfer_2026-08-16 \
  --replicates 1 --trials 12 --source-maze-training-episodes 8 \
  --target-maze-training-episodes 4 --maze-evaluation-episodes 2 --maze-steps 20
```

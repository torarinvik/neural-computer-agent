# Executable-file fast-cell transfer

This audit validates the new `ExternalProgramFastCell` seam in isolation. A
memory-side codec learns to decode values written into one source cell. The
codec is then frozen, a fresh target file receives only new opaque
action/outcome updates, and a matched fresh codec is trained online as the
control. The stream uses a fixed opaque action codebook across new logical
lifetimes so the control measures interface learning rather than an
ill-posed one-shot inverse of arbitrary continuous targets.

Run the short pressure test with:

```bash
PYTHONPATH=src:. uv run python -m experiments.external_program_fast_cell.train_transfer \
  --seed 69316 --report-out /tmp/external-program-fast-cell-69316.json
```

The test checks source retention, failed and missing-evidence no-write
behavior, persistence, frozen codec parameters, and a fresh-target learning
curve. The default 256/256 run completes in a few seconds on CPU. A positive
result is only an interface-prior transfer result: it does not show that the
frozen interpreter can invent a new procedure. The next audit must connect a
trained codec to a rendered Brain Workshop family and compare inherited versus
fresh target acquisition under complete-prefix retention, shuffled-outcome,
route-switch, and zero-replay controls.

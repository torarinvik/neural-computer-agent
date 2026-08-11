# External executable-file fast-cell transfer — 2026-08-11

This promoted bounded audit tests the CPU-plus-files memory seam. A
memory-side codec learns to decode opaque action values written into one
source cell. The codec is frozen, fresh target logical files receive new
outcome-gated writes, and a matched fresh codec sees the same target stream as
the control. The controller, interpreter, query path, value path, and cell
plasticity parameters are frozen during inherited target use.

| seed | source retention minimum | inherited stable prefix | fresh-control stable prefix | transfer ratio | promoted |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 69316 | 0.9954 | 1 | 130 | 130.00x | yes |
| 69317 | 1.0000 | 1 | 116 | 116.00x | yes |

Both seeds pass source mastery, target mastery, fresh-control measurement,
positive transfer, failed-outcome no-write, missing-evidence no-write, exact
persistence, and frozen-cell gates. The primary inherited target consumes no
optimizer updates and replays zero examples. The matched control reuses the
target exposure only as a separately reported comparator.

The action codebook is opaque and fixed across fresh logical lifetimes. It
makes the control a learnable interface-transfer test rather than an
ill-posed one-shot inverse of arbitrary continuous targets. This is a real
architectural gain: reusable memory-side computation can be acquired once and
applied to newly allocated file state while the core remains frozen.

The claim boundary remains deliberately narrow. This does not demonstrate
arbitrary new computation, unrestricted memory growth, Brain Workshop
mastery, or general continual learning. The next audit must connect the cell
codec to rendered Brain Workshop lifetimes and require complete-prefix
retention, shuffled outcomes, route switching, missing evidence, corruption,
and zero-replay transfer across genuinely new sequence rules.

Reproduce with:

```bash
PYTHONPATH=src:. uv run python -m experiments.external_program_fast_cell.train_transfer \
  --seed 69316 \
  --report-out /tmp/external-program-fast-cell-69316.json
```

Reports:

- `report_seed_69316.json`
- `report_seed_69317.json`
- `sample_efficiency_ledger.json`

# Causal repeated depth growth — 2026-08-11

This audit extends the promoted two-file growth result to three causal
working-memory files. It learns n-back-2, appends n-back-3, then appends
n-back-4. Each new file is a fresh `ExternalWorkingMemoryCell`; the shared
controller and all earlier files are frozen before the next acquisition.
Rendered cue events `4`, `5`, and `6` route the opaque files through the
external context ledger.

| seed | n-back-2 retention | n-back-3 retention | n-back-4 retention | routed accuracy | shuffled-cue target selection |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.3523 / 0.4489 / 0.2273 |
| 18 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.4034 / 0.4631 / 0.2074 |

Both seeds pass every gate: complete-prefix retention, protected-prefix
digests, frozen controller, exact cue-conditioned route separation, exact
reload with the compatible encoder, explicit rejection of an incompatible
encoder representation, isolated reversal, shuffled-cue controls, and zero
replay. Each seed used 64 source updates plus 256 updates for each appended
file, with no replayed training examples.

This promotes repeated bounded rule growth, not general continual learning.
The cells still receive a fixed rendered cue family, and the experiment does
not establish arbitrary rule induction, open-ended memory growth, compression,
or broad held-out rule transfer. The next pressure test is a held-out rule
family with cue variation and representation migration rather than another
fixed n-back depth.

Reproduce one seed with:

```bash
PYTHONPATH=src uv run python -m experiments.brainworkshop_canonical.causal_depth_growth \
  --seed 17 \
  --report-out /tmp/causal-depth-growth-17.json
```

Reports:

- `report_seed_17.json`
- `report_seed_18.json`
- `sample_efficiency_ledger.json`

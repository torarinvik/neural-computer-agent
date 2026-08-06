# Two-step external capability growth (2026-08-06)

This promoted audit extends the frozen three-program external artifact bank
through two protected sequential appends: `rotate4`, followed by
`adjacent_xor4`. Each append learns a fresh external recurrent capability,
grows the artifact bank transactionally only after the existing rows are
protected, and trains only the new route extension. The parent controller,
base router, and first route extension remain frozen while the second append
is learned.

| Gate | Seed 69316 | Seed 69317 |
| --- | ---: | ---: |
| old route retained | 1.000 | 1.000 |
| first append route | 1.000 | 1.000 |
| second append route | 1.000 | 1.000 |
| combined route | 1.000 | 1.000 |
| candidate permutation accuracy | 1.000 | 1.000 |
| reward-shuffled first/second selection | 0.000 / 0.000 | 0.000 / 0.000 |
| minimum selected behavior | 0.8008 | 0.9258 |
| route and behavior reload | pass | pass |
| corruption rejection | pass | pass |
| frozen parent/base router | pass | pass |
| replayed examples | 0 | 0 |

Both seeds passed every declared promotion gate. The result demonstrates
bounded, replay-free external growth across two append boundaries with
causal route selection and persistent state. It does not demonstrate general
continual learning, unrestricted memory growth, arbitrary new computation,
open-ended task discovery, or program induction.

Reports are in `report_seed69316.json` and `report_seed69317.json`; the
accounting summary is in `sample_efficiency_ledger.json`.

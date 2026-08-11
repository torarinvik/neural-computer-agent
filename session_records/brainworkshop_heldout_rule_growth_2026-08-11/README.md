# Held-out external rule growth

This archive contains the two-seed promotion audit for
`experiments.brainworkshop_canonical.heldout_rule_growth`.

The audit trains n-back-2/3/4 external files, trains an n-back-5 file under
rendered cue `7`, then introduces cue `8` without a route record. The route
ledger must discover the correct opaque slot from scalar verifier outcomes.

Both seeds passed every gate:

- all prefix and target retention probes: `1.0000`;
- held-out cue absent before discovery and learned from outcomes;
- held-out route recovery: `1.0000` accuracy and `1.0000` target-slot selection;
- controller and learned frontend unchanged;
- exact compatible route reload;
- incompatible learned-event representation rejected;
- zero replayed examples.

This is promoted as bounded outcome-only route discovery over bounded external
rule growth. It is not evidence of arbitrary new computation, unrestricted
memory growth, compression, or general continual learning.

Reports:

- `seed17.json` — 832 optimizer updates, 270,336 training verifier bits,
  24,864 audit bits, 0 replayed examples.
- `seed18.json` — same accounting; wall time differs by host scheduling.

`SHA256SUMS` records the report checksums.

# Cross-family rule growth — 2026-08-11

This archive contains the two-seed promotion audit for
`experiments.brainworkshop_canonical.cross_family_rule_growth`.

The audit acquires four isolated external files under one fixed learned event
and intention boundary: n-back-2, pair parity, adjacent switching, and
single-symbol parity. The final family is trained under cue `7`; cue `8` is
withheld from the route ledger and must be discovered from scalar outcomes.

Both seeds passed every gate:

- complete-prefix retention and new-family mastery;
- held-out cue absent before discovery and learned from outcomes;
- held-out route recovery with target-slot selection `1.0000`;
- frozen controller and event encoder;
- exact compatible route reload;
- incompatible learned-event representation rejected;
- shuffled-cue control; and
- zero replayed examples.

The audit uses route failure patience `1` during evidence collection and `4`
during exploitation. This is an external memory-side stability policy that
prevents one noisy outcome from abandoning a competent route.

This promotes cross-family outcome-only route discovery over bounded external
rule growth. It is not evidence of arbitrary new computation, unrestricted
memory growth, compression, or general continual learning.

Reports:

- `seed17.json` — 832 optimizer updates, 344,064 training verifier bits,
  59,648 audit bits, 0 replayed examples.
- `seed18.json` — same accounting; wall time differs by host scheduling.

`sample_efficiency_ledger.json` records the promoted accounting and
`SHA256SUMS` records report and ledger checksums.

# Three-stage append-only learned screen boundary (2026-08-07)

This audit extends the frozen append-only learned candidate screen from two
to three sequential singleton stages. The ten-candidate opaque-rule pressure
test has seven mastered base candidates and three outcome-unseen candidates,
one per isolated extension. Each later extension is activation-gated by the
cumulative scalar verifier failures of the base and all earlier stages. The
controller and mastered base screen remain frozen; no examples are replayed.

At 64 calibration updates per stage, both seeds fail the strict unseen
acquisition gate (`0.3333/0.6667`). At 128, one seed passes and one fails
(`0.8125/0.7396`). At 256, both seeds pass all gates with `1.0000` unseen
routing, `1.0000` known-context routing, exact stage-local permutations,
reload, frozen-core, reward-shuffled, and zero-replay controls.

The matched fresh-initialization control also passes both seeds at 256. The
result therefore promotes replicated three-stage bounded append growth, but
does not claim that the selective query-side prior reduces the three-stage
budget. The prior's positive efficiency result remains limited to the
two-stage mixed `[1, 2]` boundary. The current bottleneck is stage-wise
calibration/sample efficiency as sequential depth increases, not basic
capacity growth. This remains a bounded memory-side routing audit, not
general continual learning, unrestricted memory growth, or arbitrary new
computation.

Full raw reports, accounting, and checksums are included here. The `query64`
and `query128` reports are retained as decisive rejection/boundary controls;
`query256` is the promoted selective-prior run and `fresh256` is its matched
control.

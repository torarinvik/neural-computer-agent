# Replay-free partial-stream factored acquisition

This archive records a five-seed acquisition pressure test in which four
nonlinear opaque transition regimes arrived one row at a time. Each regime
had `14` current stream rows and `4` independent held-out rows. The factored
router staged a candidate after only `7` rows; later current rows updated only
that isolated external candidate. The shared base, controller, and context
encoder were frozen, and no old-regime rows were replayed.

All five seeds passed independent held-out promotion, prior-regime retention,
row-seven staging, later-candidate updates, alternating full-bundle routing,
one-row partial reads, contradictory-evidence rejection, empty-evidence
no-op, and read-only digest stability. The final stable routes were
`[0, 1, 2, 3]`; alternating revisits were
`[0, 1, 2, 3, 0, 3, 1, 2]`.

This promotes bounded replay-free partial-stream factual acquisition using a
fixed random-feature basis. It does not establish learned open-world context
formation, unrestricted memory growth, arbitrary computation, or general
continual learning. The earlier trainable-context short-prefix identity rung
remains rejected; this result isolates the improvement to one-pass factual
adaptation and safer admission, with the context encoder still frozen.

# Physical blank-program Position 1-Back promotion (2026-08-14)

Status: **promoted for bounded two-cell physical Position 1-Back acquisition**.

A controller pretrained across 160 independently projected visual frontend
families was frozen before Brain Workshop. The task-specific external temporal
address program began uniformly at `[0.25, 0.25, 0.25, 0.25]`; no learned
1-back address prior was copied into it. The learner received only captured RGB
events, acted through the ordinary position-match key, and updated the external
program only from visible green/red/blue feedback.

The first fresh physical pilot learned from 16 verifier bits: full Brain
Workshop scores changed from `75`, `71`, to `100`, while the probability of the
correct previous-event address reached `0.9379`. A second fresh program then ran
14 logical lifetimes. It captured all `168/168` stimulus onsets, emitted
`168/168` actions, and finished with `81/86 = 0.9419` cumulative public-feedback
accuracy and `1.0000` over the final rolling 44 verifier bits. The first full
rolling-44 window was already `0.8864`, so stable bits-to-0.80 is `44`. Brain
Workshop's discarded private diagnostic scores were
`62, 75, 100, 100, 100, 87, 100, 100, 100, 100, 100, 100, 100, 100`.

The controller digest remained
`59c9ef2b235104e4f0d6bc143ba195fb57a907da9f29b1d5750c39fa22f7687c`.
Exactly 86 unique public verifier outcomes caused 86 external-program optimizer
updates, zero controller optimizer updates, and zero replay. The final correct
address probability was `0.99564`. A measured retention lifetime had live tick
latency p50/p99 of `45.3/116.9 ms`; three deadlines were exceeded without losing
an onset or action.

Across 32 fresh rendered seeds, blank-program learning reached the stable
rolling-44 gate on `32/32` seeds. Blank-frozen, reward-shuffled,
action-reversed, and missing-history controls reached it on `0/32`. The best
matched fresh end-to-end learner reached it on `22/32` seeds with median 102
bits, versus `32/32` and median 44 bits for the pretrained frozen controller:
`2.32x` fewer verifier bits among successful runs.

The inherited-address run is retained only as an execution/transfer control;
it is not counted as task learning. This record does not claim dual n-back,
more than two position cells, general program induction, or unrestricted
working-memory mastery. Brain Workshop's stats file was read only by a
discarded diagnostic probe and never entered learner state or weights.

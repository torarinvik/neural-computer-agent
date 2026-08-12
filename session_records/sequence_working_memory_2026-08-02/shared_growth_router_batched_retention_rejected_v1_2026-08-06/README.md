# Batched shared-router retention: rejected (2026-08-06)

This experiment tested one batched teacher snapshot and margin-retention loss
inside each shared external route expansion. The intended benefit was to
preserve already-mastered routes while the same router acquired new routes,
without per-capability teacher modules or replay.

The prototype is rejected. With the promoted trajectory-statistics query,
hidden width 256, random candidate keys, four shifts (`6 -> 8 -> 10 -> 12 ->
14`), and 8,192 shared updates per shift, seed 69316 produced exactly the
same route-selection curve as the existing hidden-256 control:

```text
control: 0.984375, 0.968750, 0.921875, 0.812500
batched: 0.984375, 0.968750, 0.921875, 0.812500
```

The batched mechanism added approximately 181 seconds of wall time
(`170.35 s -> 351.39 s`) and did not repair the existing reversal-safety
failure. It therefore provides no measured acquisition or retention gain and
is not promoted into the experiment harness. The second seed was not run
because the first replicate did not improve the frontier; this is a decisive
rejection under the repository's one-variable promotion discipline.

Accounting for the tested arm: 4,292,984 unique verifier bits, 75,128 unique
logical lifetimes, 73,728 optimizer updates, zero replayed examples, and
351.39 seconds wall time. The report remains bounded continual-memory growth,
not general continual learning.

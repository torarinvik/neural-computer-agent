# Raw-evidence-free streaming transition candidates — promoted bounded mechanism

Across seeds `1801` and `1802`, three nonlinear transition regimes were
staged and promoted through the external router. Each regime consumed `64`
transition rows once through a fixed random-feature sufficient-statistics
model. No provisional raw rows were retained, no old committed rows were
replayed, and the frozen controller remained byte-stable.

Held-out errors were `[0.003073, 0.000021, 0.002236]` for seed `1801` and
`[0.001973, 0.000053, 0.013429]` for seed `1802`, all below the `0.02`
promotion threshold. Shuffled-next-state controls reached `1.646` and
`0.915`, respectively, and were rejected. Candidate payload restoration and
protected-slot retention passed in both seeds.

This promotes a raw-evidence-free provisional candidate boundary for bounded
nonlinear sufficient-statistics models. It does not establish replay-free
learning for arbitrary nonlinear models, unrestricted memory growth, or
general continual learning.

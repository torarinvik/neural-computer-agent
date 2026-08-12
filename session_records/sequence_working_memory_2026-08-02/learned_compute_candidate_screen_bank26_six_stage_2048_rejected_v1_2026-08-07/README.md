# Bank-26 six-stage 2048-update control rejected (2026-08-07)

This control doubles the source-screen budget again, from 1024 to 2048
updates, while keeping the 26-candidate/six-stage/full-prior configuration
fixed. All twelve unseen candidates remain perfect on both seeds, but source
known-target mastery does not replicate: seed `69316` remains at
`0.8958` aggregate with per-target holes at `0.7143`, `0.2857`, and `0.5714`;
seed `69317` passes at `0.9896`.

More source updates therefore do not repair the boundary. The remaining
failure is representation/interference limited, not a simple optimizer-budget
shortage. No general continual-learning claim follows.

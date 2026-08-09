# Replay-free nonlinear feature memory — promoted bounded mechanism

Across seeds `1401` and `1402`, the fixed nonlinear feature memory consumed
`64` opaque transition rows once and evaluated `64` held-out rows without raw
row storage, optimizer updates, or replay. Held-out errors were `0.0013821`
and `0.0014438`; both seeds passed the `0.02` verifier threshold and exact
payload restoration.

This promotes a bounded nonlinear sufficient-statistics mechanism. It does
not establish general continual learning, unrestricted computation, or
distribution-shift robustness. Reports are protected by `SHA256SUMS`.

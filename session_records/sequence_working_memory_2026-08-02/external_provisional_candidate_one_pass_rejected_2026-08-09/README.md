# One-pass provisional candidate learning — rejected

This audit disables candidate evidence-window replay. Each provisional model
receives only the current staged observation bundle, with a persistent
optimizer and the controller frozen. The committed bank remains copy-on-write
and the source slot is digest-protected.

Both seeds failed the held-out promotion gate:

| seed | held-out error | tolerance | old-evidence replay | result |
| --- | ---: | ---: | ---: | --- |
| 70611 | 0.882 | 0.2 | 0 | rejected |
| 70612 | 0.778 | 0.2 | 0 | rejected |

Isolation, persistence, and source-byte stability passed. The failure is
therefore a learning/generalization failure, not catastrophic corruption.
The current MLP cannot infer the target transition rule reliably from this
one-pass sparse stream. Repeated presentation of the current bundle was
accounted separately (`400` current-bundle presentations per seed); no older
candidate evidence was replayed.

Verdict: reject replay-free capability promotion. The cumulative evidence
window remains the promoted bounded mechanism. The next work is an online
consolidation or expandable-model mechanism that preserves one-pass evidence
without requiring the full candidate window.

Reports are protected by `SHA256SUMS`.

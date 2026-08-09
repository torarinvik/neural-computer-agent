# Promoted provisional candidate promotion rung

This follow-up closes the failure recorded in the adjacent rejected audit.
The provisional candidate now retains its cumulative verified evidence window
outside the committed bank and each candidate update trains against that
window. The committed source slot remains untouched until copy-on-write
promotion passes held-out prediction and a retention probe.

Across seeds `70611` and `70612`, the frozen controller stayed unchanged, the
source slot remained byte-stable, the bank received no write before promotion,
and both candidates passed held-out prediction at tolerance `0.2` with errors
`0.129` and `0.143`. Both candidates were promoted as slot `1`, and exact
payload persistence passed.

The accounting is intentionally explicit: four unique target evidence rows
were presented `600` times while the candidate was optimized, for `596`
candidate-evidence replays. Old committed-slot replay remained zero. This is
safe provisional evidence reuse, not a claim of replay-free general learning.

The promoted result is bounded to a tiny synthetic transition fixture. It
establishes cumulative evidence-window training as a useful credit-assignment
mechanism, not unrestricted memory growth, arbitrary new computation, or
general continual learning.

Reports are protected by `SHA256SUMS`.

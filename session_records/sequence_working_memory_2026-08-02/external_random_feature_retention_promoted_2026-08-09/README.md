# Replay-free nonlinear slot retention — promoted bounded result

Across seeds `1501` and `1502`, four disjoint nonlinear transition families
were learned into isolated external slots. Each slot consumed 64 training rows
once, then all four were revisited after later slots were acquired. Every
held-out error remained below `0.02`, slot digests were unchanged, and exact
bank persistence passed. The promoted rerun used ridge `1e-4`; the rejected
`1e-5` configuration is archived separately.

This promotes bounded replay-free nonlinear slot retention with supplied
context keys. It does not establish learned unrestricted routing, arbitrary
new computation, or general continual learning.

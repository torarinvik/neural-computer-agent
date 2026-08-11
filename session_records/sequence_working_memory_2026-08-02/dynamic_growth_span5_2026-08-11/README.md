# Dynamic external growth at span 5 (2026-08-11)

This screen tests whether a frozen controller can acquire a harder working-
memory capability in isolated external state. The growth slot is recurrent,
intention-conditioned, and context-gated. The controller core is frozen during
growth; no old examples are replayed; routing and retention are measured
separately from target execution.

The seed `69311` near-threshold run improved span-5 execution from `65.3%`
with feed-forward width-64 growth to `79.7%` with dynamic width-128 growth
after 256 updates per stage. Spans 2--4 were `100.0%`, `95.8%`, and `87.5%`.
At 512 updates, span 5 regressed to `66.3%`, demonstrating instability rather
than a simple data-scaling law.

The matched independent seed `69312` retained `100%` routing but reached only
`65.3%` on span 5 and `79.7%` on span 4. It is retained as a negative
replication, and the dynamic-growth configuration is not promoted as stable
span-5 mastery.

The evidence boundary is: dynamic external state is a promising computation
substrate, while reliable outcome-trained acquisition across seeds remains
the bottleneck. This does not establish unrestricted memory growth, arbitrary
new computation, or general continual learning.

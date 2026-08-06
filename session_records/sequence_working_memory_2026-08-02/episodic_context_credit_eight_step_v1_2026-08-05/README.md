# Eight-step replay-free episodic credit

The four-step rung passed, but its four-token pattern bank was still a finite
toy limit. This audit switches to ten same-statistics temporal patterns of
length five and acquires eight new families (`2` through `9`) after freezing
the old context and route. Each new family receives isolated external route
and credit state; later families must first attempt every earlier extension.

Across seeds 69316 and 69317, old-route accuracy, pooled-baseline separation,
candidate permutation, all eight new routes, old-route retention, and old/new
credit accuracy were `1.000`. Required extensions were attempted at `1.000`,
disabling each required extension reduced selection to `0.000`,
reward-shuffled extensions selected at `0.000`, and replay remained zero. Each
run used `286,720` unique verifier bits, `62,464` logical lifetimes, and
`4,352` optimizer updates.

The short-budget control failed old-route retention at `0.500` on both seeds;
the promoted schedule therefore increases context and route training while
keeping each new external credit head at 128 fresh updates. This records a
real sample-efficiency bottleneck rather than hiding it.

This promotes bounded eight-step replay-free external growth with isolated
episodic credit state. It does not establish unbounded memory growth, learned
consolidation, arbitrary program induction, or general continual learning.
Evidence is in `report_seed69316.json`, `report_seed69317.json`, and
`rejected_short_budget_control.json`.

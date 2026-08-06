# Generated-pattern eight-step episodic credit

This audit removes the remaining fixed pattern-bank assumption. A generated
bank enumerates all same-statistics binary procedures of length six: 20
possible patterns with three active positions. After the base families 0/1
are learned and frozen, eight new families (`2..9`) are acquired sequentially
with isolated route and event-credit state.

Across seeds 69316 and 69317, old-route accuracy, pooled-baseline separation,
candidate permutation, all eight new routes, old-route retention, and isolated
old/new credit accuracy were `1.000`. Required prior extensions were attempted
at `1.000`; required-extension ablations and reward-shuffled extensions
selected at `0.000`; and replay remained zero. Each seed used `393,216`
unique verifier bits, `75,776` logical lifetimes, and `5,632` optimizer
updates.

The under-budget control is retained separately: seed 69316 collapsed the old
context route to `0.500` while seed 69317 passed. The promoted protocol scales
context and route acquisition with episode length (`1,024` context updates,
`1,024` route updates) while keeping each new external credit head at 128
fresh updates. This is a measured robustness requirement, not hidden replay.

This promotes generated-pattern bounded eight-step replay-free external growth
with isolated episodic credit state. It does not establish unbounded memory
growth, learned consolidation, arbitrary program induction, or general
continual learning. Evidence is in `report_seed69316.json`,
`report_seed69317.json`, and `rejected_underbudget_control.json`.

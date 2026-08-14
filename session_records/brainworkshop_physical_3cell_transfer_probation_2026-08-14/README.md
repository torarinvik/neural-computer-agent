# Physical three-cell transfer probation (2026-08-14)

Status: **replicated sub-minute signal; not yet promoted**.

The mastered two-cell Position 1-Back external program was evaluated read-only
after changing exactly one frontend difficulty axis: Brain Workshop sampled
three publicly visible grid cells instead of two. Background music remained
muted, and ordinary timing, scoring, screen input, and key output were
unchanged.

Two fresh 12-trial sessions each exposed all three visual event clusters. The
agent captured `24/24` onsets, emitted `24/24` actions, and agreed with the
visible 1-back sequence on `24/24` decisions. The public verifier returned
`11/11` positive outcomes, and Brain Workshop's discarded diagnostic scores
were `100, 100`. Controller and program digests remained unchanged with zero
optimizer updates and zero replay.

The two sessions measured tick latency at `44.2/116.5 ms` and `44.9/120.9 ms`
p50/p99, with two missed deadlines each and no lost events. This is sufficient
to justify a roughly three-minute read-only three-cell retention run. It is not
sufficient to promote three cells or advance to four.

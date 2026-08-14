# Physical temporal-program bank lifecycle (2026-08-14)

Status: **promoted for one bounded two-cell Position 1-Back program lifecycle**.

A fresh external temporal-address program began uniformly while the pretrained
controller remained frozen. The first three-cell sub-minute attempt scored
`0.50, 0.33, 1.00` across lifetimes and was correctly rejected; no bank file
was created and its provisional weights were discarded. Reducing the single
curriculum axis to two visible cells produced lifetime scores `0.50, 0.67,
1.00, 1.00, 1.00, 1.00`. The first cumulative public-outcome prefix that
remained at or above 0.80 was bit 25. After 37 unique outcomes, 37 program
updates, six lifetimes, 84.33 seconds, zero controller updates, and zero replay,
the candidate cleared a stable suffix of four perfect lifetimes and entered
immutable slot 0.

Four fresh GUI sessions then instantiated the controller without the training
checkpoint. Each withheld actions for three public learned-event pulses,
selected slot 0 from the checksummed bank, and activated it for deterministic
read-only execution. The sessions emitted 36 actions after 12 warm-up events
and returned `3/3`, `2/3`, `4/4`, and `4/5` positive public outcomes: 13/15
overall. Cumulative measured retrieval accuracy was `1.000, 0.833, 0.900,
0.867`, remaining at or above 0.80. The first context was unknown before its
attempt and learned from reward; the next three were recognized before
execution. Only 15 external route observations changed. The controller digest
remained `59c9ef2b235104e4f0d6bc143ba195fb57a907da9f29b1d5750c39fa22f7687c`,
and the immutable program artifact digest remained
`90e20193a50fdfa22b75fe722e6a9e131d9ba05d7f7e7d0aedbce9fc1f3c5749`.

The instrumented final retrieval measured live tick latency at 44.4 ms p50 and
118.6 ms p99 with two deadline misses, while still capturing 12/12 stimulus
events. The curated bank file SHA-256 is
`dfe8088572678a17ca1bd919a1f33a5a58eec1a42a271134eb619e7428b37335`.

This result validates provisional learning, stable admission, checksummed
persistence, reward-conditioned route learning, retrieval, and immutable
execution for a bank containing one program. A one-slot choice has no program
discrimination burden. It does not establish multi-program routing, visible
rule-cue grounding, dual n-back, or unrestricted program induction.

After promotion, the retrieval runtime was migrated from a manual
end-of-session route update to the generic variable-port `INPUT` instruction.
A disposable copy of the curated bank received five causal reward inputs live,
scored 4/5, and learned the new context while preserving the controller and
program artifact. The instrumented p50/p99 tick latency was 45.2/127.5 ms.

# Replay-free wait statistics — promoted 2026-08-09

This archive records the two-seed pressure test for the frozen-controller
transport boundary. `EventWaitStatistics` learned a bounded age/coverage wait
policy from scalar outcomes using one-pass ridge sufficient statistics.

Both seeds consumed 192 training outcomes, 96 held-out outcomes, and 8
post-training retention outcomes, with zero optimizer updates, zero replayed
examples, and zero raw feature rows retained. The learned policy waited for a
delayed partner at age one, released a permanently absent partner at age two,
released complete windows immediately, survived a new late-absence
observation, persisted exactly, and failed the block-shuffled-outcome control.

This is a narrow promoted capability: bounded age/coverage delay and absence
handling on timestamp-buffer features. It does not establish natural temporal
inference, arbitrary missing-stream reasoning, positive transfer against a
fresh learner, or general continual learning.

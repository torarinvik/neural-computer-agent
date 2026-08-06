# Outcome-only executable-view routing — 2026-08-05

Status: promoted narrow learned memory-side routing boundary.

Two independently acquired growth procedures were compacted into one physical
artifact row with two opaque aliases/views. A separate factorized router then
learned which view to activate from controller-produced query tensors, opaque
candidate keys, attempted-view outcomes, and scalar verifier outcomes. The
router received no span labels, task IDs, correct unattempted choices, or
semantic key fields.

Both 512-update seeds passed:

- learned route accuracy: `1.000/1.000`
- candidate permutation accuracy: `1.000/1.000`
- reward-shuffled route accuracy: `0.438/0.500`
- selected views: `0` for the first procedure and `1` for the second
- wrong-view causal separation: passed for both procedures and both seeds
- one physical row with two opaque views: passed
- reload route/behavior and exact candidate state: passed
- checksum corruption rejected; frozen-core digest unchanged
- replayed examples: `0`

The 64-update pilot is retained as a curriculum rejection: routing was already
perfect, but the undertrained second artifact did not yet make wrong-view
behavior causal. Increasing acquisition updates was the only change.

This promotes learned routing of already-acquired executable views. It does
not establish arbitrary task discovery, unbounded program induction, or
general continual learning. The next pressure test is routing across more
than two views while adding new capabilities without replay.

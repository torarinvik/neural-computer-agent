# Delayed outcome-only amodal frontier

The promoted seed 37 uses a fixed four-way hidden target split across two
partial arrivals: stream `a` at tick 0, stream `b` at tick 1, an opaque action
at tick 2, and scalar reward plus exact action propensity at tick 3.

The 2,048-update run used 524,288 unique logical lifetimes, 1,048,576 unique
verifier bits, zero replayed examples, and 133.997 seconds of CPU wall time.
It reached fused reward `1.0`, missing-second reward `0.5029`, shuffled-partner
reward `0.4932`, contradictory-evidence reward `0.0`, and random-action reward
`0.2524`. The stable threshold was counted at 884,736 verifier bits after a
temporary late-training dip.

The optional memory diagnostic filled all 64 memory rows, but clearing memory
changed fused reward by only `-0.0015`. It is retained as a negative result:
this task is solvable without persistent recall, so no memory capability claim
is made.

The curated delayed checkpoint is
`artifacts/checkpoints/async_delayed_outcome_amodal_seed37.pt`.

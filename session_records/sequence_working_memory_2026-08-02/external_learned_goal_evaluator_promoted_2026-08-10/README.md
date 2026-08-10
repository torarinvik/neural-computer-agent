# Learned opaque goal evaluator — promoted

This four-seed audit extends the universal-goal pressure test with a learned
external goal verifier. `ExternalGoalEvaluator` was trained from deterministic
graded scalar verifier outcomes on nine coarse goal values with noisy state
and goal tensors. The evaluator then faced `24` held-out goal values with
additional noise while a frozen factual transition model derived behavior by
search.

All four seeds passed held-out verifier and deployed planning gates. Learned
goal mastery was `0.992`, `0.992`, `0.983`, and `1.000`; held-out positive
probabilities were at least `0.999` and held-out negative probabilities were
below `0.091`. Goal-shuffled mastery was `0.0` for every seed, reward-shuffled
evaluator mastery was `0.008`/`0.033`/`0.017`/`0.042`, and corrupted-goal
mastery was `0.0`. The controller and factual model stayed frozen during
search, evaluator persistence was exact, and all transition rows were
consumed once.

This promotion includes an important correction to the search boundary. Hard
binary goal verification is sufficient for terminal checking but provides no
intermediate gradient for long-horizon beam search. Graded scalar verifier
outcomes supply a usable goal-progress signal without adding a task policy.

The evaluator itself used `648` training rows repeatedly for `1,000` offline
optimizer updates (`647,352` repeated rows per seed), so this is explicitly
not replay-free evaluator learning. The result promotes held-out noisy goal
verification and external goal-conditioned planning, not cross-modal goal
abstraction or general continual learning. The next rung must replace the
repeated verifier batch with a one-pass/sufficient-statistics goal memory and
test representation migration without replay.

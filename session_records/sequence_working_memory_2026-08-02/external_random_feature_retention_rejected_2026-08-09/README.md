# Replay-free nonlinear slot retention — rejected configuration

This configuration trained four isolated nonlinear sufficient-statistics slots
with the default ridge value `1e-5`. Slot retention, zero replay, and exact
bank persistence passed, but capability promotion failed: held-out errors
included `0.0575258` and `0.0963505` against the `0.02` floor.

The failure is retained as evidence. It motivated one controlled change only:
increase the sufficient-statistics ridge to `1e-4` and rerun the same fixture.

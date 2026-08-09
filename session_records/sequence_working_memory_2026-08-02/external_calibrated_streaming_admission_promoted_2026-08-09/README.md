# Calibrated replay-free streaming admission — promoted

Across seeds `2001` and `2002`, a frozen transition-evidence evaluator and
independently persisted contextual calibration were connected to the
streaming candidate router. Two nonlinear transition regimes were learned
through one-pass random-feature statistics while the controller remained
unchanged. Each candidate consumed `64` rows and retained zero raw rows.

The causal control adds noise with raw MSE `0.00791` and `0.01138`, both below
the router's `0.02` continuation tolerance. Learned calibration assigns clean
evidence probabilities `0.989`/`0.906` and noisy probabilities
`0.248`/`0.179`; the corrupted stream is rejected at capacity rather than
updating either candidate. Held-out errors remain below `0.0045`, and
calibration, router, controller-freeze, persistence, and zero-target-replay
gates pass on both seeds.

This promotes a bounded learned reliability gate at the external streaming
boundary. The evaluator pretraining uses replay and is accounted for
separately; the target candidate and calibration updates use unique rows.
This is not general continual learning or learned delay compensation.

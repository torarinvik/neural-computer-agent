# One-pass evidence admission — promoted

Across seeds `2101` and `2102`, an external error-bin sufficient-statistics
evaluator learned scalar reliability from `512` unique verifier outcomes. It
performed no optimizer updates, replayed no examples, and retained only
positive/negative counts by factual prediction-error bin.

The router then acquired two sequential nonlinear regimes through one-pass
random-feature statistics. Each candidate consumed `64` rows and retained
zero raw rows. Clean evidence probability was `0.992` in both seeds; noisy
probability was `0.252` and `0.343`. The corrupted control had raw MSE
`0.00565` and `0.00439`, below the router tolerance of `0.02`, but was
rejected without candidate modification. Held-out errors stayed below
`0.0084`; persistence, frozen-controller, candidate-isolation, and zero-replay
gates passed.

This promotes replay-free sufficient-statistics reliability at the external
streaming boundary. It does not establish interleaved nonlinear admission,
learned delay/absence handling, positive transfer against fresh learners, or
general continual learning.

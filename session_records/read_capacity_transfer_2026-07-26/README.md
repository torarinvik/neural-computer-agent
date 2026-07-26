# Gradual compute-transfer rung — pre-registration

The direct memory-read → recurrent-thought jump failed twice. This rung changes
only one environmental axis: the external-memory bank capacity changes from
three to four. The learner is not told the capacity.

The inherited 105-parameter advantage head races an identical reset head on
the same 720 fresh attempted-action verifier bits and 12 updates. Reward,
feature, and missing-evidence controls begin from inherited weights. No
unattempted outcome, correct action, correct answer, task identity, or capacity
label is learner-visible.

Pass requires the inherited head to maintain at least 65% choice accuracy,
beat the strongest fixed policy by 0.05 utility, capture 20% of the oracle gap,
and cross all three thresholds stably before reset. Controls must cost at least
0.02 utility; old tasks, gradients, and persistence must pass. One pass permits
one unchanged replication. Failure closes this representation even on a near
neighbor and does not earn more experience.

## Initial result and precision diagnostic

Seed 7821 passed from zero new bits while reset first crossed at 240. Seed 7822
retained a positive inherited utility advantage over reset at every prefix, but
missed the fixed 65% choice gate (`62.9%`) and therefore rejected formal
replication.

The 256-context audit has roughly three percentage points of binomial standard
error near the decision boundary. Before changing the mechanism, both
deterministic training runs are repeated with 2,048 private test contexts.
Training remains exactly 720 bits and 12 updates. These are measurement
diagnostics only: they cannot retroactively pass the original replication.
Agreement under the powered audit would justify a newly pre-registered
replication; disagreement closes the rung.

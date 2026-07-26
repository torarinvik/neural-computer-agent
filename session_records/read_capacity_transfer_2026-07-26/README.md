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

Both powered diagnostics agreed:

- seed 7821: inherited stable at 0 bits, reset at 240; final gap capture
  `55.9%` versus `42.7%`;
- seed 7822: inherited stable at 0 bits, reset at 600; final gap capture
  `48.0%` versus `39.2%`.

Every causal, persistence, gradient, and retention gate passed. Therefore seed
7823 is pre-registered as the first formal powered run, with the same 2,048
test contexts, 720 training bits, and all original thresholds. A pass permits
one unchanged seed-7824 replication. These fresh seeds, not the diagnostics,
decide promotion.

## Replicated promoted result

Both fresh, pre-registered powered seeds passed:

- seed 7823: inherited stable mastery at `0` new bits versus reset at `120`;
  final choice accuracy `72.1%`, utility gain `+0.1922`, and oracle-gap capture
  `54.0%`;
- seed 7824: inherited stable mastery at `0` versus reset at `120`; final
  choice accuracy `71.1%`, utility gain `+0.1965`, and gap capture `50.7%`.

Reward-shuffled, feature-shuffled, and missing-evidence controls lost the
inherited benefit. Persistence, gradients, binary mapping, and four-rule
retention passed in both runs. Each learner used 720 fresh verifier bits and
12 updates; each powered private audit used 4,096 counterfactual bits.

This promotes a narrow but real compounding result: a compute allocator learned
at capacity three immediately masters capacity four, while a matched reset
learner requires 120 new verified outcomes. The gain is learning speed, not
exclusive final capability.

The next rung should change one additional generic-statistics axis—such as a
held-out occupancy/reliability range or capacity five—while keeping the
attempted-action objective fixed. Only after that bridge replicates should the
system revisit a different physical operation such as recurrent thought.

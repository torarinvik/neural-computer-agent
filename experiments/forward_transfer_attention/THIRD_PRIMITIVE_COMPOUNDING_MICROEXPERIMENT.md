# Third-primitive predictive-curriculum test

## Question

Does adding correctly paired spatial experience to an already temporal-trained
predictive core reduce the reward bits needed for a third, distinct delayed
same/different identity primitive?

This tests whether the reusable predictive core improves as its experience
widens. It is not by itself proof that verified mastery of the spatial
behavior caused the gain, because the spatial reward head does not update the
frozen core.

## Third primitive

Each lifetime contains two sequential RGB object frames. The objects either
share an identity or have different identities, exactly balanced by a private
verifier. Positions, backgrounds, palette pairs, and surface identities vary.
The learner outputs through a new seed-specific four-command protocol.

Training uses only frozen recurrent state, attempted command, known uniform
propensity, and observed scalar reward. Same/different labels and unattempted
command outcomes remain private.

## Prior-experience factorial

Every core starts from identical weights:

1. temporal paired prediction only;
2. temporal paired prediction, then spatial paired prediction;
3. temporal paired prediction, then the same spatial pixels with future
   targets shuffled across lifetimes;
4. temporal paired prediction, then an equal amount of additional temporal
   paired experience;
5. fully fresh core.

All second-stage pretraining arms use equal lifetimes, steps, batches, and
examples. All phase-C action learners use fresh intentions/adapters and equal
reward-bit prefixes, updates, and examples.

## Metrics and audits

- AULC above the 50% verified majority floor;
- reward bits to 55%, 65%, and 75%;
- paired spatial versus shuffled-spatial and extra-temporal thresholds;
- true identity counterfactual: change only the second object's identity and
  recompute the same/different label;
- missing-first and missing-second evidence;
- opposite-rule stale state;
- protocol swap;
- action-shuffled and reward-shuffled training controls.

## Promotion gate

The temporal+spatial paired curriculum advances only if it:

- beats both equal-compute spatial-shuffled and extra-temporal controls by at
  least 0.03 AULC;
- reaches a fixed threshold with fewer phase-C reward bits;
- reaches at least 60% final held-out accuracy;
- reaches at least 60% counterfactual accuracy with at least 50% prediction
  flips;
- returns to at most 55% when either required identity frame is removed;
- falls to at most 40% under opposite-rule stale state.

One seed is provisional; three are required for an exploratory second
cross-primitive transition. Even two successful transitions are insufficient
for a compounding slope; the six-primitive ledger remains the standard.

## Seed-211 result (bounded negative)

The same/different capability learned causally, but the curriculum gate failed:

- temporal + paired spatial experience: 80.47% final, 0.2422 AULC, 75% at 256
  reward bits;
- temporal + spatial future shuffled: 81.77%, 0.2578 AULC, 75% at 128 bits;
- temporal + equal extra temporal experience: 82.03%, 0.2568 AULC, 75% at 128
  bits;
- temporal only: 82.03%, 0.2406 AULC, 75% at 128 bits;
- fully fresh: 50.00%, 0.0000 AULC.

The candidate's AULC advantage was -0.0156 and it required twice the reward
bits of both equal-compute controls to reach 75%. No replication seeds were
run.

The negative is not a failed task or shortcut. Changing only the second
identity yielded 84.64% relabeled accuracy and 65.10% prediction flips.
Removing the first or second identity returned to 49.22%/50.26%, while
opposite-rule stale state and protocol swapping each reduced accuracy to
19.53%.

Within this budget, adding spatial predictive training after temporal training
caused no compounding gain and is consistent with representational drift or
task-irrelevant specialization. A rehearsal/retention intervention or a task
whose actions change future observations is more justified than adding more
spatial-only steps.

# Operation-aligned re-query curriculum — pre-registration

## Question

Does preserving a learned second-ranked re-query policy reduce the verified
experience needed when memory capacity increases from five to six?

This is the smallest nearby difficulty shift available: the physical operation,
action meaning, cost, sensory/controller path, and four generic evidence
features remain unchanged. Only the number of competing latent memories grows.

## Discovery race

First reconstruct seed 7893 exactly and save its learned re-query head. Then run
one 720-bit discovery race at capacity six:

- **re-query lineage:** the complete learned capacity-five re-query head;
- **read ancestor:** the earlier capacity-five read-value trunk with its
  operation-specific output reset;
- **reset:** a newly initialized head;
- **causal controls:** reward shuffled, evidence shuffled, and evidence absent.

All learners receive only attempted actions and their scalar verifier outcomes.
The unattempted outcome, correct compute choice, answer, and semantic task
identity remain private.

## Gate

The re-query lineage must:

1. stably reach at least 65% compute-choice accuracy;
2. beat the strongest fixed policy by at least 0.03 verified utility;
3. capture at least 20% of the available oracle gap;
4. reach stable mastery in strictly fewer unique verifier bits than both the
   read ancestor and reset;
5. retain the binary-mapping and four-rule skills;
6. pass evidence, reward, gradient, and exact-persistence controls.

Only a complete pass permits an unchanged replication seed. A tie is not
compounding evidence. A failure or regression closes capacity six at this
curriculum step; it does not authorize a longer run.

# Procedural shape two-query curriculum bridge

## Breakthrough

The controller learned a verified second sequential query over the previously
mastered ordinal set. This isolates query-history complexity from the still
unsolved third-ordinal-after-query frontier.

| run | stable target bits | final accuracy | weakest populated cell | span-3 q1 retention | span-2 q2 retention |
|---|---:|---:|---:|---:|---:|
| primary seed 29431 | 5,376 | 98.19% | 97.07% | 99.02% | 99.17% |
| replica seed 30431 | 3,072 | 98.23% | 96.81% | 98.44% | 98.84% |

The strict audit now reports every query-position × presented-ordinal cell.
At this curriculum rung, query position two contains ordinals one and two;
ordinal three is intentionally withheld rather than averaged away.

## Causal controls

- Blank presentation remained at chance: 49.89% primary, 50.05% replica.
- Full active and workspace reset remained at chance: 49.76% primary,
  50.13% replica.
- Valid pixel-level reversal caused predictions to change on 97.99% and
  98.10% of verifier-changed cases.
- Candidate rerender caused predictions to change on 97.46% and 97.88%.
- Shuffling scalar outcomes prevented mastery and drove final accuracy down
  to 69.60%.

## Localization and rejected paths

Frozen-state probes showed that the exact second-query reader input retained
enough information for 97.1% supervised action decoding. Nevertheless:

- training the inherited relation adapter for about 100 relevant updates
  stayed flat at 54.8% on the second-query/third-ordinal cell;
- modest whole-controller plasticity stayed near chance on that cell;
- a fresh zero-output direct action residual also stayed near chance;
- making the third item redundant with item one did not solve the cell.

The failure is therefore a conditional query-history frontier, not missing
sensory evidence, raw storage capacity, or simple adapter interference.

The adjacent curriculum at frontier 0.0 mastered quickly. Introducing 5% of
the third-ordinal second-query cases produced the first non-flat signal,
reaching 62.75% in the final audit while retaining old skills. At 10%, the
rare cell was too sample-starved and noisy to establish further progress.

## Next frontier

Preserve the successful history bridge while increasing the proportion of
third-ordinal second queries. The next experiment should prioritize those
rare verified outcomes in the loss or sampler so gradual difficulty does not
also mean vanishing learning signal.


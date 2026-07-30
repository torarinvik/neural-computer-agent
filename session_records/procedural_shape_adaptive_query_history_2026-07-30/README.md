# Adaptive query-history curriculum

## Result

The controller now stores three independently rendered shapes and answers two
arbitrarily ordered visual queries without receiving game state or semantic
labels. The successful curriculum separated two operations:

1. repeat the same lookup after one answer;
2. retrieve the third item after a different lookup.

Difficulty advanced only after every populated query-position × ordinal cell
remained above 95%. The second seed learned more slowly, so the scheduler held
both rungs longer rather than increasing difficulty on a fixed timetable.

| lineage | repeated-query final | crossed-query final | hard crossed cell | span-3 q1 retention | span-2 q2 retention |
|---|---:|---:|---:|---:|---:|
| primary | 99.76% | 99.41% | 97.80% | 99.87% | 99.69% |
| replica | 99.41% | 99.27% | 98.00% | 99.61% | 99.35% |

All values are held-out. Missing-presentation and all-memory-reset controls
remain at chance.

## Compounding evidence

Before the repeat-query rung, the second-query/third-ordinal cell was about
50%. After mastering only repeated lookup, but before receiving crossed-query
training, the same fully crossed cell rose to 72.17%. This is direct forward
transfer from the simpler working-memory operation.

The earlier direct full-difficulty adapter experiment remained at 54.79%
after 38,400 target feedback bits. From the repeated-query checkpoint, focused
crossed-history training produced a smooth primary curve:

74.6% → 76.2% → 80.6% → 83.1% → 87.6% → 91.9% → 95.3% → 96.4%.

The independent replica followed the same pattern, but needed a longer hold:

69.7% → 73.3% → 81.5% → 86.7% → 92.3% → 94.6% → 95.5% → 96.5% → 97.5%.

## Sample-efficiency accounting

Stable target feedback bits are cumulative across continuation phases:

| lineage | repeat rung | crossed rung |
|---|---:|---:|
| primary | 53,760 | 76,800 |
| replica | 76,800 | 92,160 |

No examples were replayed. Each unique binary verifier outcome counts as one
feedback bit.

The 1% and 10% mixture experiments were rejected. Although their overall
accuracy stayed near 99%, subgroup audits showed crossed-history accuracy at
only 73–76%. Rare-case inverse-frequency weights created high-variance
gradients. The lesson is that conceptual difficulty and evidence density must
be controlled separately: introduce one new operation at a time, but practice
that operation often enough to learn it.

## Causal controls

- Primary hard crossed subgroup: 97.80%.
- Replica hard crossed subgroup: 98.00%.
- Primary valid reversal flip rate: 99.02%.
- Primary candidate-rerender flip rate: 98.90%.
- Reward-shuffled training collapsed to 51.24%; hard subgroup 53.03%.
- Primary blank presentation: 50.04%; all-memory reset: 50.13%.
- Replica blank presentation: 49.98%; all-memory reset: 50.38%.

## Next frontier

The next adjacent operation is a third sequential query. Start by repeating a
previously answered lookup, then introduce a different third lookup only after
stable mastery. Query-history subgroup gates must remain separate so aggregate
accuracy cannot hide one failed position.


# Three-query history curriculum

## Result

The same controller now stores three independently rendered shapes and answers
three sequential visual queries. It receives pixels, its own opaque actions,
and scalar verifier outcomes only. No shape identity, ordinal, correct action,
or game state is learner-visible.

The curriculum introduced exactly one operation at each rung:

1. answer a third query by immediately repeating query two;
2. answer a third query by retrieving query one after an intervening query;
3. answer a third query about the remaining, previously unqueried item.

Difficulty advanced only after every populated query-position × presented-
ordinal cell remained above 95% at every later measured checkpoint.

| lineage | immediate bits | delayed bits | novel bits | cumulative bits | final overall | hard q3 cell |
|---|---:|---:|---:|---:|---:|---:|
| primary | 5,760 | 17,280 | 5,760 | 28,800 | 99.70% | 98.78% |
| replica | 17,280 | 23,040 | 5,760 | 46,080 | 99.45% | 98.00% |

Each bit is one unique binary verifier outcome. No examples were replayed.

## Compounding evidence

The weakest third-query cell on the primary parent was:

- 85.50% for immediate repetition;
- 81.88% for delayed repetition;
- 80.37% for a novel third lookup.

After learning immediate repetition, delayed repetition rose zero-shot to
90.87%. After learning delayed repetition, the novel lookup rose zero-shot to
92.33%. The replica reproduced the same final transition: novel lookup began
at 93.46%, then reached stable mastery after only 5,760 additional verifier
bits.

This is forward transfer, not merely a higher final score: each adjacent
primitive supplied most of the next primitive before the next rung received
training.

## Retention

The new behavior did not erase the old behavior:

| lineage | one-query retention | two-query retention |
|---|---:|---:|
| primary | 99.87% | 99.84% |
| replica | 99.69% | 99.81% |

The retained tasks also preserve their causal signatures: blank presentation
and full memory reset remain at chance, while valid reversal and candidate
rerender flip rates remain above 99% or within normal sampling variation.

## Adversarial controls

- Primary final blank presentation: 50.04%.
- Primary final all-memory reset: 49.96%.
- Primary valid reversal flip rate: 99.43%.
- Primary candidate-rerender flip rate: 99.40%.
- Replica final blank presentation: 50.02%.
- Replica final all-memory reset: 49.85%.
- Replica valid reversal flip rate: 99.33%.
- Replica candidate-rerender flip rate: 99.10%.
- Matched reward-shuffled training never reached the stable gate and collapsed
  to 55.33% overall.

The shuffled run initially inherited strong behavior, then destroyed it as
incorrect outcomes accumulated. Correct lifetime-level experience is therefore
causally necessary for the acquired skill.

## Interpretation

This is a verified short-term-memory lookup primitive, not yet general working
memory. The controller can retain three visual identities and use the retained
sequence for three sequential equality queries. It has not yet demonstrated
arbitrary sequence transformation, reordering, arithmetic, planning, or
learned dynamic allocation of memory capacity.

## Next frontier

Keep the sequence length fixed at three and introduce one minimal manipulation:
return the immediately preceding item, then the item two positions back, before
attempting reverse-order recall. Advancement should remain adaptive and should
be scored by stable verifier bits, forward transfer, and retention—not by a
fixed training schedule.

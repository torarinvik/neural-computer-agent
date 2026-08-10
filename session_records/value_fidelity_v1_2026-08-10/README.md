# Value fidelity: polarity asymmetry (F112)

Probe 212. `game_slots.py` with the F111 configuration
(`--forage-twins --train-updates 40000 --horizon 4 --balance-loss
--seek 0.85 --ignorance 0.5 --freeze-objects --value-head`) plus
`value_fidelity()`: score every cell with the value head, compare
against the verifier's ground-truth worth map. 2 seeds (69316, 69317),
12 held-out worlds each.

## Result (pooled)

| measure | pooled | normal | inverted (~) |
| --- | ---: | ---: | ---: |
| predicted-vs-truth correlation | 0.1727 | 0.1694 | 0.1761 |
| top cell is food | 0.1198 | 0.2188 | 0.0208 |
| top cell is poison | 0.0156 | 0.0182 | 0.0130 |

Behaviour in the same runs: held-out +0.0010 / +0.0127 (floor -0.0470),
twin -0.0901 / -0.1034, withheld ~= floor — the F111 result reproduces.

## Reading

- Low global correlation coexists with 45.6% behavioural headroom
  because beam search needs local ranking, not a faithful map.
- Poison avoidance transfers across polarity; food attraction does not:
  on inverted worlds the argmax cell is almost always empty. The entry
  suppresses the flagged object but cannot promote the newly-edible one.
- Correlation identical across polarity while top=food differs 10x:
  the defect is at the extreme of the ranking, not the bulk.

Next fix target: a mechanism that can promote, not only suppress —
signed entry interaction in the value head, or diff-entries.

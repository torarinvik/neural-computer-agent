# The exception store closes the last gap (F97)

F96 predicted: "`walled` goes 0.894 -> ~1.000 by storing 27 exceptions, with the
rule-bank untouched." Built and measured. Plant frozen, entry unchanged, zero
gradient steps; exceptions recorded only where the rule is observably wrong.

| family | rule only | watch 128 | watch 256 | watch 512 | watch 1024 |
| --- | ---: | ---: | ---: | ---: | ---: |
| walled | 0.894 | 0.928 (8) | 0.965 (18) | 0.984 (23) | **1.000 (27)** |
| grid | 1.000 | 1.000 (0) | 1.000 (0) | 1.000 (0) | 1.000 (0) |
| toggle | 0.992 | 0.994 (0) | 0.995 (1) | 0.999 (2) | 0.997 (2) |
| dial | 0.980 | 0.980 (2) | 0.981 (4) | 0.982 (8) | 0.986 (18) |
| perm | 1.000 | 1.000 (0) | 1.000 (0) | 1.000 (0) | 1.000 (0) |
| line | 1.000 | 1.000 (0) | 1.000 (0) | 1.000 (0) | 1.000 (0) |

accuracy (exceptions stored). Both seeds land on exactly 27 for `walled` —
precisely the number of transitions on which grid and walled differ.

## The degeneracy check

A store that fixes everything by memorising everything is a lookup table wearing
an architecture. This one holds ZERO entries for grid, perm and line at every
observation budget. It grows only where rules fail.

## Limit is observation, not capacity

8 -> 18 -> 23 -> 27 as the world is watched longer. 128 random draws from 256
possible pairs cover ~39%, so the intermediate figures are what sampling
predicts. Nothing is learned; the system looks.

## Scope

The store is an exact dict keyed by (state, action) — an idealised
content-addressed memory. A learned/approximate store would be lossy and this
does not speak for it. What is established: the information needed is small,
precisely localised, and obtainable by watching. Failure mode to watch: a family
whose rule captures little would grow the store toward the whole table. `dial`
at 18 is the closest instance; store size per family is the diagnostic.

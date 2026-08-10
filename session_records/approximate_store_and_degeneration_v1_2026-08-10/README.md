# Approximate store, and the degeneration case (F98)

F97's exception store was an exact dict keyed by (state, action) — idealised.
This tests a realistic similarity-addressed store with capacity, and a family
with no rule at all.

| family | rule | exact dict | mean-pooled key | concatenated key |
| --- | ---: | ---: | ---: | ---: |
| walled | 0.894 | 0.996 | 0.908 | 0.996 |
| toggle | 0.992 | 0.999 | 0.951 | 1.000 |
| dial | 0.980 | 0.985 | 0.968 | 0.986 |
| chaos | 0.014 | 0.986 | 0.557 | 0.977 |
| grid/perm/line | 1.000 | 1.000 | 1.000 | 1.000 |

Mean-pooling slot embeddings loses state identity, so the store misses and
mis-fires — on `toggle` it was NET-HARMFUL (0.992 -> 0.951). Concatenating slot
embeddings preserves identity and matches the exact dict everywhere.

Content addressing is fine for exceptions; LOSSY addressing is not.

## Degeneration is real

`chaos` has a random permutation table — no rule exists. Rule alone 0.014; the
store grows to 249/256 entries and reaches 0.986. The store DOES become the
whole table when no rule exists.

## The guard: violation rate, not capacity

| family | violation rate | verdict |
| --- | ---: | --- |
| grid, perm, line | 0.0% | rule holds |
| toggle | 1.1% | rule holds |
| dial | 2.1% | rule holds |
| walled | 10.3% | rule + exceptions |
| chaos | 98.5% | NO RULE — refuse to memorise |

Two orders of magnitude of separation. A capacity cap alone is a poor guard:
bounding chaos to 32 entries holds memory down but accuracy collapses to 0.127 —
silent failure. Violation rate says WHY the store is growing.

# Opaque basis compatibility-prior audit — 2026-08-08

This audit trains a replaceable memory-side compatibility screen from scalar
verifier outcomes over opaque instruction queries and opaque basis signatures.
On held-out queries, the learned screen orders candidates before fresh
verification. It cannot change whether a query is admissible; it can only
reduce trial count when at least one candidate clears the verifier threshold.

| seed | admissible queries | learned trials | cold trials | learned pass rate |
| --- | ---: | ---: | ---: | ---: |
| 69316 | 237 | 1.034 | 1.869 | 0.926 |
| 69317 | 241 | 1.037 | 1.979 | 0.941 |

Both seeds preserved verifier admissibility and reduced trials on admissible
queries with zero replayed examples. This promotes screening efficiency on a
bounded scalar-verifier audit, not verifier-free admission or general
cross-operator transfer.

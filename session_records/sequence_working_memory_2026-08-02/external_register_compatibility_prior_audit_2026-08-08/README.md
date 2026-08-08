# Opaque basis compatibility-prior audit — 2026-08-08

This audit trains a replaceable memory-side compatibility screen from scalar
verifier outcomes over opaque instruction queries and opaque basis signatures.
On held-out queries, the learned screen orders candidates before fresh
verification. It cannot change whether a query is admissible; it can only
reduce trial count when at least one candidate clears the verifier threshold.

| seed | admissible queries | learned trials | cold trials | learned pass rate |
| --- | ---: | ---: | ---: | ---: |
| 69316 | 242 | 1.116 | 2.074 | 0.945 |
| 69317 | 244 | 1.029 | 2.143 | 0.953 |

Both seeds preserved verifier admissibility and reduced trials on admissible
queries with zero replayed examples. This promotes screening efficiency on a
bounded scalar-verifier audit, not verifier-free admission or general
cross-operator transfer.

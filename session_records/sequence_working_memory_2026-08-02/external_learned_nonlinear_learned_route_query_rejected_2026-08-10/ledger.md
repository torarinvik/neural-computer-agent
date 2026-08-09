# Experiment ledger

| field | value |
| --- | --- |
| objective | Replay-free continual identity routing for nonlinear factual memory |
| seeds | 82601, 82602, 82603 |
| unique verifier bits | 512 per seed |
| unique logical lifetimes | 512 per seed |
| controller optimizer updates | 0 |
| context-encoder optimizer updates | 0 |
| factual model optimizer updates | 192 per seed |
| route-scorer optimizer updates | 384 per seed |
| route-scorer unique current windows | 3 per seed |
| route-scorer current-window reuses | 381 per seed |
| old-regime replay | 0 |
| raw provisional rows retained | 0 |
| factual acceptance | strict `0.01` match tolerance |
| result | rejected: shared route scorer forgot earlier identities |
| controls | partial evidence, corruption rejection, safe factual fallback, frozen controller, exact persistence |

The report JSON files are the authoritative per-seed records.

# Experiment ledger

| field | value |
| --- | --- |
| objective | Retain route identity in isolated external memory without replay |
| seeds | 82601, 82602, 82603 |
| unique verifier bits | 512 per seed |
| unique logical lifetimes | 512 per seed |
| controller optimizer updates | 0 |
| context-encoder optimizer updates | 0 |
| factual model optimizer updates | 192 per seed |
| route-memory optimizer updates | 0 |
| route-memory state updates | 8, 15, 19 |
| route-memory replayed examples | 0 |
| old-regime replay | 0 |
| raw provisional rows retained | 0 |
| factual acceptance | strict `0.01` match tolerance |
| result | rejected as a route-capability gain; exact external-memory retention passed |
| controls | partial evidence, corruption rejection, safe factual fallback, frozen controller, exact persistence |

The report JSON files are the authoritative per-seed records.

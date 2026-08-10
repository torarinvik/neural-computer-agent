# Experiment ledger

| field | value |
| --- | --- |
| objective | Retain and revisit learned nonlinear factual models without replay |
| seeds | 82601, 82602, 82603 |
| regimes | 4 nonlinear opaque regimes per seed |
| rows | 48 presented of 64 train rows; 64 held-out rows per regime |
| current-window gradient steps | 16 per four-row bundle |
| unique verifier bits | 512 per seed |
| unique logical lifetimes | 512 per seed |
| factual model optimizer updates | 768 per seed |
| address-adapter optimizer updates | 208 per seed |
| current-window reused updates | 720 per seed |
| controller optimizer updates | 0 |
| context-encoder optimizer updates | 0 |
| replayed examples | 0 |
| old-regime replay | 0 |
| raw provisional rows retained | 0 |
| route-query updates | 0 |
| result | promoted bounded factual-memory retention |

The report JSON files are the authoritative per-seed records.

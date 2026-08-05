# Producer→consumer global-parity replication — 2026-08-04

Seed `69105` independently reproduced the sequential composition result:
composed accuracy was `89.58%`, while parent, producer-only, and
consumer-only controls were `49.09%`, `50.00%`, and `48.70%`. Producer and
prior-read ablations returned to chance, and the frozen controller core and
artifact reload gates passed.

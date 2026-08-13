# Composite goals (F222)

Goal = sum of up to two signed distance terms; greedy search (singles
then pairs of top-4). 3 seeds x 10 worlds.

intercept1 POSITIVE for the first time since F203: +0.703/+0.469/+0.812
per seed (single-term: negative on all). composite - single on
intercept: +0.7656 +- 0.1977, t=+3.87. Mechanism: recurring cross-axis
term (1,2)->(3,0) = lead pursuit, invented by the search. Registered
prediction said intercept would NOT move — refuted in reverse; the
prediction that choice WOULD be fixed is also refuted (t=+1.00, and
bank>oracle persists on choice and intercept: composition widens
expression without closing the model-proxy gap).

Reproduce: python -m experiments.games_amodal.probes.composite --seed S

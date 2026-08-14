# F229: depth frontier closes; proxies at depth 1 are the optimum
# (2026-08-13)

ev/  evaluator_planning.py, 6 seeds: ridge transition-utility
     evaluator (held R2 0.87-1.0), eval_d2 > eval_d1 (+0.335,
     t=+2.68) but dominated by the proxy stack (-0.70, t=-8.9).
hc/  horizon_ceiling.py, 6 seeds: privileged exhaustive true-return
     search depths 1-4. collect2 proxy +2.57 vs truth-d4 +0.63;
     headroom localized to pursue compounds (+0.19..+0.33).
es/  evaluator_shaped.py, 6 seeds: potential-based synthesis
     (lambda=0.05 registered); shaped_d2 - cost_d1 = -0.31 (t=-4.36).

Narrative and prediction ledger: docs/MEMORY_BANK_DESIGN.md F229.
Scope: fixed on this family. Next program: sealed mechanism benchmark.

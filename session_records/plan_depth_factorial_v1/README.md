# F228: objective inadequacy localized (2026-08-13)

pdf/ plan_depth_factorial.py, 6 seeds: sel-depth x exec-depth x
     bank/oracle, leaf scoring (cumulative cells archived in
     pursuit_and_plan_depth_v1/pd).
rf/  rank_fidelity.py, 6 seeds: per-state bank-vs-oracle agreement,
     leaf-cost correlation, best-action agreement vs true return,
     regret, decisive fraction. Uses pd-archived goals.
tr/  truereturn_arm.py, 6 seeds: privileged oracle arms — cumulative
     cost, true-return depth 1 and depth 2 (diagnostic ceiling).

Verdict (registered readings): true-return lookahead +0.143 t=+3.78;
state-cost lookahead fails under both dynamics (bank t=-4.40, oracle
t=-3.66); depth-2 selection does not rescue. The goal representation
is the failing layer. Narrative: docs/MEMORY_BANK_DESIGN.md F228.
Scope: fixed on this family.

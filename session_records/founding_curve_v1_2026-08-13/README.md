# The founding curve (F219)

3 seeds x {k=0,6,12,18} x 6 held eval worlds. Warm = read+repair
(programs) and read+beam (goals); cold = full searches. Self-labelled
experience only.

    k    prog w/c        goal w/c    ret warm   ret cold
    0    2720/3584       4.0/173     -0.428     +0.688
    6    1370/3584       3.2/173     +0.545     +0.688
    12   1313/3584       3.6/173     +0.603     +0.688
    18   1398/3584       3.3/173     +0.615     +0.688

~24x total acquisition cost reduction at k=18 (rollout = 320 steps),
warm-cold deficit -0.0729 t=-1.07 (ns; k=12 was -0.0851 t=-2.15).
Curve saturates at k=6 (family coverage). k=0 competence collapse
(-0.428, below random) shows cost cuts without experience are worthless.
Shuffled-label control at k=18: 2360-2814 candidates vs 1088-1614
experienced vs 3348-3762 cold.

Reproduce: python -m experiments.games_amodal.probes.founding_curve
--seed S --reader-updates 4000. Single-threaded.

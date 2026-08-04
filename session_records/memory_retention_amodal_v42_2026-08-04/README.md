# Long-horizon retention instability rejection

v42 used the parent-preserving v41 protocol for a requested 2,048-update
retention phase. It maintained 1.0 mastered-primitive retention and reached
near-perfect final retention, but later held-out validation prefixes oscillated
and the stable-prefix gate remained unsatisfied. The run is rejected as a
long-horizon promotion despite strong final snapshots.

The failure motivates v43's explicit three-consecutive-validation early-stop
consolidation policy. Raw results are in `seed_19.json`; no checkpoint or
long-horizon capability claim is promoted from v42.

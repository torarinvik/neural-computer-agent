# Batch-variance control rejection

This mini-rung changed only the trajectory batch size from 16 to 64. The
parent stabilized, but intact recall and both order controls stayed at chance:
`0.490` target-first and `0.493` target-last. More trajectories per update do
not solve the retention failure, so batch-size scaling is rejected as the
immediate intervention.

# Fresh-initialization transfer-gate qualification

v49 repeats the seed-19 transfer lead with three independent fresh learner
initializations, identical unseen tokens, verifier seed, and transfer budget.
The transferred learner reaches stable threshold at 13,312 bits. Only one
of three fresh learners qualifies its parent and reaches 20,480 stable bits;
the other two fail parent qualification and never enter retention training.

The transfer status is fresh_parent_not_qualified, and no population transfer
ratio is claimed. The earlier 1.538x result is retained as a single favorable
fresh-control comparison only. The new report schema makes this failure
explicit instead of treating it as a missing scalar.

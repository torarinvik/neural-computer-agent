# v61 token-diverse retention control

The fixed two-token curriculum was a likely lookup shortcut. v61 varies the
opaque event-token pair across episodes while preserving the same token for
each episode's write and recall. Full randomization gives strong unseen-token
recall for seeds 18 and 19 but leaves seed 17 parent acquisition at chance.

The matched fixed-parent/random-retention schedule fixes that retained-model
variance: all three retained models reach `0.996–1.000` unseen recall and pass
the narrow causal/persistence gates. Transfer remains unqualified at the
512-update fresh rung: only seed 17 qualifies. The token-diverse retention
curriculum is promoted as a narrow training control, not as a population
transfer claim or checkpoint.

# Learned WAIT/COMMIT follow-up

This is the follow-up to the rejected 256-update deliberation rung. The run
uses 4,096 observable transport-schedule warmup lifetimes to stabilize the
opaque action path, then freezes that path and trains the execution head for
4,096 fresh scalar-outcome lifetimes. The execution head is still part of the
same recurrent controller; the warmup schedule uses only whether the generic
partner event is present, never the hidden target or correct action.

Across seeds 17, 18, and 19, the learned policy commits on complete windows,
waits on delayed windows, reaches 1.0 reward, and beats both immediate commit
and fixed waiting on mixed utility. This promotes the first learned
`WAIT`/`COMMIT` compute-allocation capability. Mixed-state `THINK` arbitration
was not promoted: the combined distribution still needs a state-conditioned
execution critic or curriculum. The isolated think-required primitive is
recorded separately under
`session_records/deliberation_think_required_2026-08-03/`.

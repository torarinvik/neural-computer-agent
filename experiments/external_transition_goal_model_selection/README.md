# Goal-reachability selection over factual external models

This audit gives a bounded external bank several independently learned
factual transition models with different opaque dynamics. A current opaque
goal and candidate intention are supplied to model-based search; the selector
chooses the stable model whose predicted rollout reaches that goal best.

This is the model-over-policy strategy from the exported session: behavior is
derived at inference time from reusable transition facts. The controller is
not updated, no task policy is stored, and transition evidence is not replayed
by the selector. Promotion requires randomized goals, model order, stable
logical addresses, and a held-out goal-verifier margin over random selection.

Both seeds reached `1.000` selection versus `0.333` random. Evidence is
archived under
`session_records/sequence_working_memory_2026-08-02/external_transition_goal_model_selection_promoted_2026-08-09/`.

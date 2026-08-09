# Goal-reachability selection over factual models — promoted

This two-seed audit selected among three independently learned factual
transition models by rolling each model forward under the current opaque goal
and candidate intention. The selected stable logical address was the model
with the lowest predicted goal distance.

Both seeds achieved `1.000` selection accuracy versus `0.333` random, with a
positive held-out goal margin on every evaluation. The controller was not
updated, no task policy was stored, and no transition examples were replayed.

This promotes bounded model-over-policy goal selection. It does not establish
general continual learning, unrestricted growth, or robustness to arbitrary
representation drift; those remain the next pressure tests.

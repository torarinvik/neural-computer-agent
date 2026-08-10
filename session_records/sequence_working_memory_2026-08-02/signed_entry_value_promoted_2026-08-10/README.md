# Promoted signed external-entry value model

Three seeds (`9101`, `9102`, `9103`) pass the polarity-flip audit. The model
learns a positive state-only salience from a source stream containing only
positive opaque entries. With the model frozen and zero target updates, a
held-out negative entry reverses the value prediction with mean squared errors
of `1.82e-4`, `2.05e-5`, and `2.10e-5`. The matched unfactorized control has
target errors of `0.0352`, `0.0476`, and `0.0632`.

All seeds pass the mixed-polarity entry-shuffle causal control, exact oddness,
neutral zero-entry behavior, source retention, exact persistence, frozen
parameters, and zero replay. The entry is therefore the isolated polarity
delta; the shared state salience does not need to be repainted for the
contradictory target.

This promotes a reusable signed-delta value boundary, not arbitrary value
learning, general continual learning, or unrestricted memory growth. It is a
mechanistic bridge toward the exported session's “replace the puzzle piece”
architecture and still needs integration with live model-based search on a
nontrivial multimodal stream.

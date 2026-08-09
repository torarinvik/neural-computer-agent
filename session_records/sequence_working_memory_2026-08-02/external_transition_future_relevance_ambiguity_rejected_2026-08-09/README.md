# Matched-telemetry future-relevance ambiguity — decisively rejected

This two-seed control used four real factual transition models with equalized
bank-owned usage, age, and prediction-error telemetry. Two unprotected models
were otherwise indistinguishable; a hidden future schedule randomly selected
which one had to be retained. The lifetime policy received only the current
bank state and one verifier outcome per proposal.

Held-out selection was `0.460` and `0.595` against a `0.500` random ceiling.
Exact policy persistence, zero controller updates, and zero replay gates
passed. The result is intentionally not promoted: it shows that generic
lifetime telemetry cannot infer future relevance when the present evidence is
matched. Adding more usage/age/error heuristics would be the wrong direction.

The architectural implication is a goal-conditioned external query/relevance
boundary. The query must be learned from the current opaque event, intention,
and/or goal state; it must remain separate from raw modality formats and from
the controller's replaceable weights. The verifier still decides whether a
retention or eviction transaction commits.

# Partial and ambiguous open-world factual memory — promoted

This three-seed pressure test combines online address formation, nonlinear
factual dynamics, partial evidence, and bounded ambiguity handling. The
context encoder starts untrained and receives zero optimizer updates. Four
nonlinear regimes arrive through `32/64` presented transition rows; the
external address adapter learns isolated copy-on-write identities while the
controller remains frozen.

Two novel regimes are staged concurrently. An eight-row bundle whose factual
predictions are deliberately indistinguishable is quarantined outside the
committed bank. Later clearly routed evidence anchors the bundle to one opaque
candidate, and the quarantined rows are consumed once through streaming
sufficient statistics. The stream then grows the bank, revisits all regimes in
alternating order, and runs a corrupted-candidate rejection control.

| seed | regimes | max held-out MSE | quarantine rows | replay |
| ---: | ---: | ---: | ---: | ---: |
| 82501 | 4 | 0.00109 | 8 | 0 |
| 82502 | 4 | 0.00107 | 8 | 0 |
| 82503 | 4 | 0.00287 | 8 | 0 |

All gates passed on all three seeds: untrained encoder, partial evidence,
quarantine without candidate mutation, later one-time resolution, verified
capacity growth, held-out factual promotion, alternating revisits, prior-slot
retention, corruption rejection without a bank write, copy-on-write address
isolation, frozen controller, zero replay, and exact router persistence.

The new implementation rule is important: a later opaque factual-routing
decision can anchor previously quarantined evidence, and the router consumes
that evidence in the same one-pass adaptation transaction. Quarantine is not
silently treated as a model update and cannot be promoted while unresolved.

Claim boundary: bounded replay-free nonlinear factual-memory identity under
partial and explicitly ambiguous evidence. The stream and capacity are finite,
the factual basis is a fixed random-feature family, and this does not establish
unrestricted continual learning, arbitrary new computation, or learned
multimodal context formation.

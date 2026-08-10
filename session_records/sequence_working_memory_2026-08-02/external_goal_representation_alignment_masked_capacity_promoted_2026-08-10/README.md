# Promoted verifier-gated masked-prototype replacement

This four-seed audit fills one identity slot's bounded masked-prototype
capacity, rejects an unsafe replacement transaction, and then accepts a new
partial pattern only after a retention probe verifies the core affine identity,
the nonlinear identity, the prior masked route, and the new route. Updates are
selected from opaque anchors; no frontend or slot ID is supplied to the bank's
runtime update API.

All seeds passed. Rejected candidates preserved the live digest. Accepted
replacement retained both core routes and the old masked route while adding the
new masked route. Affine mastery was `1.0`; nonlinear mastery was `0.9917`–`1.0`.
Persistence was exact, the controller/model/verifier stayed frozen, and replay
was zero.

This promotes bounded verifier-gated masked-prototype replacement under fixed
capacity. It does not establish autonomous retention policy, unbounded growth,
semantic open-world identity, or general continual learning.

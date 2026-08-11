# External temporal-history memory contract

This diagnostic qualifies the storage primitive needed to move beyond a
fixed event window. `ExternalTemporalHistoryMemory` appends learned event
tensors into scoped, variable-capacity records and exposes only opaque
relative-offset reads. Sequence positions and physical records stay outside
the controller.

With width 16 and 128 records in each of two scopes, the probe stored 256
records, read offsets `0`, `1`, `7`, and `127` exactly, reloaded the payload
with zero difference, isolated a scope clear, represented missing history with
an explicit mask, and rejected a checksum-corrupted payload. No controller
parameters, verifier bits, or optimizer updates were used.

## Claim boundary

This is an ABI/storage qualification only. It does not show that the learner
can discover a useful offset, learn arbitrary addressing, or retain a new
capability without replay. The next pressure test must train an external
offset/address selector from scalar outcomes and compare it against a fresh
learner, with wrong-offset, missing-history, shuffled-outcome, and retention
controls.

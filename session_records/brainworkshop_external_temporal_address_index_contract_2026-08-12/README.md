# External temporal address-index contract

This diagnostic qualifies the replaceable content-addressed boundary between
learned opaque query keys and external temporal history. The index stores keys
and opaque `(scope, absolute_position)` locations, while the history store
owns event payloads and metadata. Absolute positions are required: a relative
offset would move an old address whenever newer records were appended.

The focused contract tests verified exact metadata-preserving reads, explicit
misses, stale/out-of-range locations, checksum rejection, reload identity, and
stable retrieval after later appends. The canonical runtime test verified that
the resolved prior token is transient, the current token remains persistent,
and an address miss reaches the controller only as `present=False` evidence.

## Claim boundary

This is a storage and runtime-interface qualification only. It uses no verifier
bits, optimizer updates, replay, learned query discovery, compression, or
general continual-learning claim. The next capability experiment must train the
opaque query key/index policy from outcome-only feedback and include fresh,
missing-address, corrupted-memory, shuffled-outcome, and retention controls.

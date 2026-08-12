# Canonical temporal-history bridge v2: transient related-key retrieval

This rerun validates related-key content retrieval after the history isolation
correction. Persistent memory stores opaque learned query keys and capability
addresses; prior history is transient processing context, and only current
events persist in the controller event window.

Across seeds `17` and `18`, all `16/16` retrieval gates passed: source and
target mastery/retention, exact and 20%-perturbed key recovery, unknown-key
rejection, clear, reload, checksum-corruption rejection, frozen
controller/frontend/capability state, and zero replay. Minimum retained
source/target accuracies were `1.0`/`1.0` and `0.921875`/`0.90625`.

Each seed used `158,208` unique verifier bits, `16,640` logical lifetimes,
`256` optimizer updates, `264` route-memory updates, two content-memory
writes, and zero replay.

This promotes corrected bounded related-key retrieval through the temporal
bridge. It does not establish learned compression, capacity management,
arbitrary new computation, unrestricted memory growth, or general continual
learning. Raw reports are `seed-17.json` and `seed-18.json`.

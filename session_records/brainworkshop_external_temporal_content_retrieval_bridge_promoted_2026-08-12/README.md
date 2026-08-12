# Canonical-bridge related-key temporal content retrieval

This promotion composes the bridge-integrated query-conditioned temporal
address capability with persistent content-addressed external memory. The
memory stores opaque learned query keys and capability-address values; the
canonical runtime bridge reads prior event tensors before appending the
current event, while the controller, event encoder, and capability file remain
frozen during retrieval.

Across independent seeds `17` and `18`, all `16/16` gates passed: source
mastery and retention, target mastery, exact and 20%-perturbed key recovery,
unknown-key rejection, clear, exact reload, checksum-corruption rejection,
frozen controller/frontend/capability state, and zero replay. The related-key
cosine scores were `0.9712`/`0.9870` for seed `17` and `0.9841`/`0.9869` for
seed `18`. Minimum retained source/target accuracies were `1.0`/`1.0` and
`0.921875`/`0.90625`.

Each seed used `158,208` unique verifier bits, `16,640` logical lifetimes,
`256` optimizer updates, `264` route-memory updates, two content-memory
writes, and zero replay.

This promotes bounded related-key retrieval through the canonical temporal
history bridge. It does not establish learned compression, capacity
management, arbitrary new computation, unrestricted memory growth, or general
continual learning. The raw reports are `seed-17.json` and `seed-18.json`.

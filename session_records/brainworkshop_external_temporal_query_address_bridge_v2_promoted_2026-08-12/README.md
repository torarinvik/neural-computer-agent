# Canonical temporal-history bridge v2: transient prior context

This rerun validates the corrected history boundary. The bridge reads prior
learned event tensors before appending the current event, places prior tokens
before the current token so latest-event semantics remain correct, and passes
only current tokens into the controller's persistent event window. The prior
tokens are transient processing context rather than duplicated controller
state.

Across seeds `17` and `18`, all `14/14` query-address gates passed: source
mastery and retention, target acquisition, logical lags `4`/`5`, unknown and
wrong-address rejection, missing-history rejection, shuffled-outcome rejection,
exact route reload, frozen readout/controller/frontend, and zero replay. Seed
`17` retained both capabilities at `1.0`; seed `18` retained minimum source
and target accuracies of `0.921875` and `0.90625`.

Each seed used `158,208` unique verifier bits, `16,640` logical lifetimes,
`256` optimizer updates, `264` route-memory updates, and zero replay.

This promotes the corrected bounded query-conditioned address boundary. It does
not establish unrestricted search, learned compression, arbitrary new
computation, or general continual learning. Raw reports are `seed-17.json` and
`seed-18.json`.

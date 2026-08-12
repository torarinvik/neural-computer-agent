# Canonical-bridge query-conditioned temporal address growth

This promotion reruns same-cue query-conditioned temporal addressing through
the production `AmodalControllerRuntime.step_streams_with_external_history()`
boundary. The bridge reads prior learned event tensors before appending the
current event, so public logical lags `4` and `5` are passed to the bridge as
relative offsets `3` and `4`. The external route table selects one opaque
logical lag per query and reuses it for the episode.

Across independent seeds `17` and `18`, all `14/14` gates passed: source
mastery and retention, target acquisition, correct lags, unknown-query and
wrong-lag rejection, missing-history rejection, shuffled-outcome rejection,
exact route reload, frozen readout/controller/frontend, and zero replay. Seed
`17` retained source and target at `1.0000`; seed `18` retained minimum source
and target accuracies of `0.921875` and `0.90625`.

Each seed used `158,208` unique verifier bits, `16,640` logical lifetimes,
`256` optimizer updates, `264` external route-memory updates, and zero replay.
The controller and event encoder were byte-stable, and the external capability
readout was frozen before target growth.

This promotes canonical temporal-history transport plus bounded
query-conditioned external addressing. It does not establish unrestricted
memory search, learned compression, arbitrary new computation, or general
continual learning. The raw reports are `seed-17.json` and `seed-18.json`.

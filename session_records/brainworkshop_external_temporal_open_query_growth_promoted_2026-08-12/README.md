# Sequential open query/address growth

This promoted bounded rung grows one frozen temporal readout from one source
route to five opaque query/address routes. New routes for depths `5`, `6`, `7`,
and `8` are acquired sequentially from fresh paired scalar verifier probes.
After every addition, every earlier route is evaluated without replay. The
canonical variable-capacity content index stores the learned context keys and
opaque route positions.

Seeds `17`, `18`, and `19` pass all `16/16` gates. Every retained prefix and
every newly acquired route reaches `1.0000`; all five 20%-related keys retrieve
correct routes at `1.0000`. Unknown-key miss, shuffled-feedback rejection,
reload, corruption rejection, clear, frozen controller/event encoder/file,
and zero replay pass.

Per seed: `169,600` unique verifier bits, `358,400` counterfactual-arm bits,
`17,088` unique logical lifetimes, `36,864` counterfactual logical lifetimes,
`1,024` optimizer updates, `264` route-memory updates, five content-memory
writes, and zero replay.

This establishes repeated bounded external address growth and prefix
retention, not unrestricted memory growth, arbitrary new computation,
compression under capacity pressure, or general continual learning. Raw
reports are `seed-17.json`, `seed-18.json`, and `seed-19.json`.

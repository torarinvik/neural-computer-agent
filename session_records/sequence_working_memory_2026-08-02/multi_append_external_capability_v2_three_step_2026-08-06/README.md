# Three-step external capability growth (2026-08-06)

Status: replicated promoted bounded-growth result.

This audit extends the frozen three-program external artifact bank through
three protected sequential appends: `rotate4`, `adjacent_xor4`, then
`complement_rotate4`. Each append learns a fresh external recurrent
capability, grows the artifact bank transactionally only after every existing
row is protected, and trains only the new route extension. The generic
append-only route chain keeps the parent controller and earlier route state
frozen while the bank grows.

Both seeds passed every declared gate:

- six executable artifacts present and protected;
- old route, all three append routes, and the combined route: `1.000`;
- candidate permutation accuracy: `1.000`;
- reward-shuffled append selection: `0.000` for every append;
- selected behaviors, seed 69316: `0.891`, `0.980`, `0.898`, `0.941`, `0.801`, `0.969`;
- selected behaviors, seed 69317: `0.926`, `0.961`, `0.984`, `0.965`, `0.930`, `0.969`;
- every wrong-artifact causal comparison passed;
- route/artifact reload and corruption rejection passed;
- parent and base-router digests remained unchanged;
- replayed examples: `0`.

For both seeds, old and append route rates were `1.000`, all reward-shuffled
append selections were `0.000`, and replayed examples were `0`. Accounting is
recorded in `sample_efficiency_ledger.json`; complete raw reports are
`report_seed69316.json` and `report_seed69317.json`. This promotes a stronger
bounded replay-free external-growth result, not general continual learning,
unrestricted memory growth, arbitrary new computation, or program induction.

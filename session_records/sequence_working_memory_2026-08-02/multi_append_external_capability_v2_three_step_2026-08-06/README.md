# Three-step external capability growth (2026-08-06)

Status: provisional promoted result pending the required second seed.

This audit extends the frozen three-program external artifact bank through
three protected sequential appends: `rotate4`, `adjacent_xor4`, then
`complement_rotate4`. Each append learns a fresh external recurrent
capability, grows the artifact bank transactionally only after every existing
row is protected, and trains only the new route extension. The generic
append-only route chain keeps the parent controller and earlier route state
frozen while the bank grows.

Seed 69316 passed every declared gate:

- six executable artifacts present and protected;
- old route, all three append routes, and the combined route: `1.000`;
- candidate permutation accuracy: `1.000`;
- reward-shuffled append selection: `0.000` for every append;
- selected behaviors: `0.891`, `0.980`, `0.898`, `0.941`, `0.801`, `0.969`;
- every wrong-artifact causal comparison passed;
- route/artifact reload and corruption rejection passed;
- parent and base-router digests remained unchanged;
- replayed examples: `0`.

Accounting for seed 69316 is recorded in `sample_efficiency_ledger.json` and
the complete raw report is `report_seed69316.json`. A second seed is required
before calling this a replicated promotion. The claim remains bounded
replay-free external growth, not general continual learning, unrestricted
memory growth, arbitrary new computation, or program induction.

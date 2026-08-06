# Unseen prefix-parity append (2026-08-06)

Status: provisional promoted result pending the required second seed.

This audit extends the replicated six-artifact route-isolated bank with a
fourth append, `prefix_parity4`, whose cumulative parity procedure was not in
the earlier append set. The controller, base router, and six earlier artifact
paths remain frozen. The new capability is learned in fresh external state,
admitted only after the existing bank is protected, and activated through the
generic append-only route chain after fresh scalar failure evidence.

Seed 69316 passed every declared gate:

- seven executable artifacts present and protected;
- old, all four append, and combined route accuracy: `1.000`;
- reward-shuffled selection: `0.000` for every append;
- selected `prefix_parity4` behavior: `0.8789`;
- every selected capability was mastered and every wrong-artifact comparison
  was causal;
- route/artifact reload and corruption rejection passed;
- parent and base-router digests remained unchanged;
- replayed examples: `0`.

The run used `141,312` unique logical lifetimes and `354,304` unique verifier
bits over `935.17` seconds. The complete raw report is
`report_seed69316.json`; the second seed is required before promoting this as
replicated unseen-computation acquisition. The claim remains bounded external
growth, not general continual learning, unrestricted memory growth, or
arbitrary program induction.

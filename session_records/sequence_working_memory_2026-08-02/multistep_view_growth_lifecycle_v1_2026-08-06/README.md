# Canonical lifecycle-backed two-step growth (2026-08-06)

This reruns the promoted two-step external-capability pressure test through
`ExternalCapabilityLifecycle`. The coordinator composes artifact storage,
retention gates, verified consolidation, and capacity-safe adoption without
interpreting the executable payload or updating the controller.

Four protected opaque views are followed by two fresh additions:
`rotate` and `complement_rotate`. Each addition is independently acquired,
behavior-verified, retained, and routed while the controller, old router, and
first route extension remain frozen. The second extension receives no replay
of the first extension's route data.

## Promoted result

Both replication seeds passed every gate:

| metric | seed 69316 | seed 69317 |
| --- | ---: | ---: |
| old route accuracy | 1.0000 | 0.9922 |
| first new route | 1.0000 | 1.0000 |
| second new route | 1.0000 | 1.0000 |
| two-step chain | 1.0000 | 0.9974 |
| candidate permutation | 1.0000 | 0.9974 |
| optimizer updates | 3,584 | 3,584 |
| unique verifier bits | 229,376 | 229,376 |
| replay after either append | 0 | 0 |

Both runs preserved protected behavior, passed causal wrong-view and
reward-shuffled controls, reloaded exact opaque views, rejected checksum
corruption, and kept the controller and first extension frozen.

## Interpretation

This promotes a reusable memory-side lifecycle contract for two sequential
protected additions. It is still bounded continual-memory scaling: the new
executable artifacts are externally trained, the route extensions are
explicit growth state, and the bank is not yet open-ended. The next blocker is
generalizing the transaction to longer nonstationary sequences while keeping
retention confidence calibrated and proving transfer against a fresh learner.

The complete machine-readable reports are `report_seed69316.json` and
`report_seed69317.json`.


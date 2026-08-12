# Balanced open-world route discovery

This record promotes a bounded external-memory acquisition result. The audit
admitted eight opaque control-flow files incrementally, then learned eight
unseen context-to-file bindings in an interleaved outcome-only stream while the
controller and all encoders remained frozen.

The first version used stochastic inverse-attempt novelty alone. That failed
for the right mechanistic reason: an unseen context had a uniform lottery over
the growing file bank, so the newest correct file could receive no trial. The
promoted fix is an external `strategy="balanced"` policy. Until a context has a
protected winner, it samples only among the least-attempted files; after
mastery, it returns to exploit-plus-novelty exploration with exact propensities.

Across seeds 17, 18, and 19, every context selected its matching file, including
the newest file. Context 7 was reversed to file 0 while the other seven context
bindings were not replayed and remained retained. Context keys stayed distinct,
and protected-file, reload, corruption, frozen-controller, zero-replay, and
shuffled-feedback controls passed. The shuffled arms did not master the target
mapping.

Accounting is 462 verifier bits and 448 logical lifetimes per arm/seed, for
2,772 verifier bits and 2,688 logical lifetimes across the three positive and
three shuffled arms. Optimizer updates and replayed examples were zero.

This is a bounded open-world route-discovery promotion. It does not establish
unrestricted memory growth, arbitrary program induction, or general continual
learning.

See `report_summary.json` and `sample_efficiency_ledger.json` for the compact
machine-readable record. The runnable audit is
`experiments/recipe_expressibility/control_flow_runtime_open_world_route_growth.py`.

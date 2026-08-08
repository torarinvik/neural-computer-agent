# Seven-capability verifier-gated adapter-sharing growth — promoted

Date: 2026-08-08

The frozen-controller protocol now acquires seven opaque capabilities without
replaying earlier examples. Fresh verifier probes select compatible adapter
sharing or choose between fresh-adapter and fresh-compute growth. Retention is
measured on a fixed held-out suite per mastered capability at every later
prefix; local recovery remains available for a newly grown capability below
threshold.

| seed | final behavior | physical compute | physical adapters | promoted |
| --- | --- | ---: | ---: | --- |
| 69316 | `1.000 / 0.895 / 1.000 / 0.797 / 0.766 / 0.813 / 0.789` | 7 | 2 | yes |
| 69317 | `1.000 / 0.750 / 0.867 / 0.855 / 0.770 / 0.832 / 0.879` | 6 | 2 | yes |

Both runs pass stable-prefix mastery, fixed-holdout retention during growth,
old-weight protection, frozen-core, exact reload, memory-corruption recovery,
reduced payload, and zero-replay gates. This is a bounded seven-capability
promotion, not general lifelong learning or unrestricted memory growth.

Reports are covered by `SHA256SUMS`.

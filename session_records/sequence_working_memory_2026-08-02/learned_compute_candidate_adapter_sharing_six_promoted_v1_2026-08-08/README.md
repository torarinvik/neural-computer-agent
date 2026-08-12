# Six-capability verifier-gated adapter-sharing growth — promoted

Date: 2026-08-08

The frozen-controller external-memory protocol acquires six opaque
capabilities without replaying earlier examples. Fresh verifier outcomes select
compatible adapter sharing or choose between fresh-adapter and fresh-compute
growth. Fixed held-out retention probes are reused at every later prefix, and
local recovery remains available for a newly grown capability below threshold.

| seed | final behavior | physical compute | physical adapters | promoted |
| --- | --- | ---: | ---: | --- |
| 69316 | `1.000 / 0.895 / 1.000 / 0.797 / 0.766 / 0.813` | 6 | 2 | yes |
| 69317 | `1.000 / 0.750 / 0.867 / 0.855 / 0.770 / 0.832` | 5 | 2 | yes |

Both runs pass stable-prefix mastery, fixed-holdout retention during growth,
old-weight protection, frozen-core, exact reload, memory-corruption recovery,
reduced payload, and zero-replay gates. This is a bounded six-capability
promotion, not general lifelong learning or unrestricted memory growth.

Reports are covered by `SHA256SUMS`.

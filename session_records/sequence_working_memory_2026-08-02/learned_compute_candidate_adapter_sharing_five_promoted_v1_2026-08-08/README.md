# Five-capability verifier-gated adapter-sharing growth — promoted

Date: 2026-08-08

The verifier-scored growth protocol now acquires five opaque capabilities with
the controller and old capabilities frozen. It first tests fresh compute
against frozen adapters, then scores fresh-adapter versus fresh-compute growth
when sharing fails. A newly grown capability may receive bounded local
recovery updates from fresh outcomes only. Retention uses one fixed held-out
probe suite per mastered capability at every later prefix, preventing random
probe variation from being misclassified as forgetting.

| seed | final behavior | physical compute | physical adapters | promoted |
| --- | --- | ---: | ---: | --- |
| 69316 | `1.000 / 0.895 / 1.000 / 0.797 / 0.766` | 5 | 2 | yes |
| 69317 | `1.000 / 0.750 / 0.867 / 0.855 / 0.770` | 4 | 2 | yes |

Both runs pass stable-prefix mastery, fixed-holdout retention during growth,
old-weight protection, frozen-core, exact reload, memory-corruption recovery,
reduced payload, and zero-replay gates. This is a bounded five-capability
promotion, not general lifelong learning or unrestricted memory growth.

Reports are covered by `SHA256SUMS`.

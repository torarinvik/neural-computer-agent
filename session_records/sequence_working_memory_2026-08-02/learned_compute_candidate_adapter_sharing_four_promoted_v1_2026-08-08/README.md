# Four-capability verifier-gated adapter-sharing growth — promoted

Date: 2026-08-08

This is the four-capability extension of the bounded frozen-controller audit.
Each new opaque capability receives fresh verifier outcomes. The memory-side
policy first tests fresh compute against frozen adapters, then scores fresh
adapter growth versus fresh compute-plus-adapter growth when sharing fails.
The optional local recovery budget is available, but neither canonical run
needed recovery at this boundary. The controller, shared base, and all old
capabilities remain frozen while each new capability is acquired.

| seed | final behavior | physical compute | physical adapters | promoted |
| --- | --- | ---: | ---: | --- |
| 69316 | `1.000 / 0.895 / 1.000 / 0.797` | 4 | 2 | yes |
| 69317 | `1.000 / 0.750 / 0.867 / 0.855` | 3 | 2 | yes |

Both runs pass stable-prefix mastery, old-weight protection, frozen-core,
exact reload, memory-corruption recovery, retention during growth, reduced
payload, and zero-replay gates. The result is a bounded four-capability
promotion, not general lifelong learning or unrestricted memory growth.

Reports are covered by `SHA256SUMS`.

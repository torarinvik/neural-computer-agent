# Stable identity confirmation diagnostic

Date: 2026-08-11
Status: **rejected as a promoted gain; retained as an opt-in verifier primitive**
Schema: `neural-computer.brainworkshop-factored-residual-sequence-pressure.v1`

## Question

Does cumulative partial routing improve when close factual ties are confirmed
by a stable winner across independent prefixes, instead of requiring every
prefix to clear the configured margin on its own?

The new `stable_identity_confirmation` policy still computes factual model
errors for every cumulative prefix. It only removes the per-prefix margin
floor; the same factual slot must win every confirmation bundle. Contradictory
evidence and reliability vetoes remain refusals, and the route is read-only.

The address update also received a correctness hardening: prefix alignment
uses fixed full-view targets and replaces slot adapters without rewriting the
historical opaque route keys. The factual model, controller, and committed
contexts remain unchanged.

## Matched results

| seeds | condition | complete | regime promotions | missing-evidence passes |
| --- | --- | ---: | ---: | ---: |
| 91–93 | baseline | 0/3 | 8/9 | 0/3 |
| 91–93 | stable confirmation | 1/3 | 8/9 | 1/3 |
| 94–96 | baseline | 1/3 | 5/9 | 1/3 |
| 94–96 | stable confirmation | 1/3 | 5/9 | 1/3 |

The combined result is only `2/6` versus `1/6` complete runs, and the fresh
seed replication did not separate the conditions. This is a directional,
seed-sensitive signal, not a promoted capability gain.

## Interpretation

The export's bind-once principle is useful at the verifier boundary, but the
current pressure test remains limited by upstream regime promotion and opaque
identity quality. The policy is safe enough to retain as an explicit opt-in
primitive because it requires cross-prefix winner stability and preserves
contradiction refusal. It must not be enabled by default or cited as general
continual learning until it improves a fresh-seed curve and survives the full
promotion gate.

## Accounting

- Seeds: `91, 92, 93, 94, 95, 96`.
- Ten-step lifetimes, three target regimes, two promotion holdouts.
- Stable `91–93`: 338 unique verifier bits, 54 logical lifetimes, 270 rows consumed once, zero replay, zero optimizer updates.
- Stable `94–96`: 266 unique verifier bits, 51 logical lifetimes, 230 rows consumed once, zero replay, zero optimizer updates.
- Controller unchanged and base frozen in every run.
- Reports and checksums are stored beside this README.

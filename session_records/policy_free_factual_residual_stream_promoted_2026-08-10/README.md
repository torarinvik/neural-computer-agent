# Promoted: multi-regime policy-free factual residual stream

Date: 2026-08-10
Seeds: `101`, `102`
Schema: `neural-computer.policy-free-factual-residual-stream.v1`

## Result

Both seeds promote six distinct factual regimes plus a reversal into seven
opaque external residual slots. The shared transition base is trained once,
then frozen. Each residual consumes `32` unique transition rows through
one-pass random-feature sufficient statistics; no residual optimizer replay is
used. Every admission passes held-out one-step, two-step recursive rollout,
and complete-prefix retention probes.

| seed | max prefix MSE | max rollout MSE | source MSE | selected codec |
| ---: | ---: | ---: | ---: | :--- |
| 101 | 0.004544 | 0.018566 | 0.004217 | `torch.float16` |
| 102 | 0.009934 | 0.016057 | 0.000352 | `torch.float16` |

The opaque route round-trip returns slots `0..6`. Novel-bundle admission makes
`21` existing-slot comparisons across the seven lifetimes. Empty and
corrupted evidence are non-mutating; shuffled reversal evidence is rejected;
exact persistence reproduces the retained-prefix losses; and the frozen base
remains byte-stable.

Float16 compression passes held-out verification and reduces residual-bank
storage from `125,552` to `62,804` bytes (`49.98%` smaller). Int4 is correctly
rejected by the same retention probe. Fresh nonlinear controls use `2,400`
optimizer updates and replay `76,800` examples, while the residual bank uses
zero replay.

This promotes bounded factual-memory scaling with verifier-gated growth and
compression. It does not establish general continual learning, arbitrary new
computation, unlimited memory growth, or policy learning.

Reports: `seed-101.json`, `seed-102.json`.

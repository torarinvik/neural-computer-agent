# Verifier-gated adapter-sharing growth — bounded promotion

Date: 2026-08-08

This promotion evaluates a frozen controller with an external reusable-compute
library. For each new opaque capability, the memory-side policy first tests a
fresh recurrent compute module against frozen existing adapters. If no adapter
passes fresh verifier probes, it scores two isolated growth operators: a fresh
adapter on protected compute, and fresh compute plus fresh adapter. Only the
best fresh-probed candidate is retained. If the selected new capability is
still below threshold, it receives 128 additional local updates using fresh
outcomes only. Rejected candidates are removed and their stochastic stream is
restored.

Canonical source order `[0, 1, 2]` promotes across both independent seeds:

| seed | final behavior | adapter-sharing result | promoted |
| --- | --- | --- | --- |
| 69316 | `1.000 / 0.895 / 1.000` | later capability reused adapter 0 | yes |
| 69317 | `1.000 / 0.750 / 0.867` | later capability reused adapter 0 | yes |

All canonical runs pass stable-prefix mastery, old-weight protection, frozen
core, exact reload, memory-corruption recovery, no-replay, and reduced physical
payload gates. The retained library has three logical bindings, two physical
compute modules, and two physical adapters in both canonical runs.

The source-order permutation control `[2, 1, 0]` also promotes: seed `69316`
reaches `1.000 / 0.820 / 0.934` without recovery, and seed `69317` reaches
`1.000 / 0.840 / 1.000` after the 128 local recovery updates. This supports
order-robustness for this bounded control, not a claim of general continual
learning.

Reports and the permutation control are covered by `SHA256SUMS`.

# Replay-free nonlinear factual-memory retention (2026-08-10)

This is the first promoted learned-MLP factual-memory rung after the
four-step current-window control failed. Each of four nonlinear opaque
regimes received `48/64` training rows. A frozen controller and frozen
context encoder were retained while each external candidate received sixteen
local optimizer updates per four-row current window. No old-regime evidence
was replayed and no raw provisional rows were retained.

| seed | held-out acquisition | revisit identity | prior retention | corruption rejection | promoted |
| ---: | :---: | :---: | :---: | :---: | :---: |
| 82601 | pass | 6/6 | pass | pass | yes |
| 82602 | pass | 6/6 | pass | pass | yes |
| 82603 | pass | 6/6 | pass | pass | yes |

The improvement is a verified factual-model capacity result, not a route
memory result. The router selected the correct factual slot by evaluating
opaque transition prediction; no route-query module or route-memory state was
needed. This is important because the separate route-prototype experiments
still mis-propose some slots. Route proposals remain an optional accelerator,
never the correctness authority.

## Interpretation

Increasing local current-window fitting from four to sixteen updates reduced
the nonlinear model's prediction error enough for factual matching and
retention to pass across all three seeds. The controller stayed frozen and
the bank remained external, append-only, and independently persistent.

The claim boundary is deliberately narrow: four synthetic nonlinear regimes,
finite capacity, supplied opaque transition bundles, and bounded model
optimization. This does not establish unrestricted growth, general
continual learning, arbitrary new computation, or learned multimodal route
formation. The next pressure test should vary the dynamics family or partial
evidence while retaining the sixteen-step accounting and matched fresh
controls.

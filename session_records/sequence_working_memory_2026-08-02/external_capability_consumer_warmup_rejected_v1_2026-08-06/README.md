# Replay-free consumer warm-up — 2026-08-06

Status: rejected as a promoted continual-learning strategy; retained as a
matched training-control diagnostic.

The experiment tested whether a downstream external consumer learns more safely
when its output decoder is trained alone for the first 64 updates, followed by
joint decoder/consumer training. The inherited `complement4` head was frozen,
downstream raw events were hidden, the shared controller stayed frozen, and
all arms used fresh verifier examples with zero replay. The no-warm-up arm used
the same seed, budgets, and random streams.

| arm | consumer accuracy | blank accuracy | reward-shuffled accuracy | consumer causal | stable consumer bits |
| --- | ---: | ---: | ---: | :---: | ---: |
| warm-up 64 | 0.6719 | 0.5703 | 0.7188 | yes | none |
| no warm-up | 0.6406 | 0.5703 | 0.5156 | no | none |

Warm-up improved the real target and restored the consumer ablation signal at
this medium rung, but it also let reward-shuffled feedback produce a larger
score than the real target. Because the control is not robustly verifier-
causal and neither arm reached a stable mastery prefix, the strategy is not
promoted. The result supports a narrower conclusion: output calibration may
help optimization, but it must be paired with stronger causal and stable-prefix
controls before it can be used for continual-learning admission.

The 16-update pilot remained at chance in every arm and is retained only to
record the failed sub-minute curriculum rung. Full accounting is in
`sample_efficiency_ledger.json`; raw reports are kept beside it.

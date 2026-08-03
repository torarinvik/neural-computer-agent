# `neural_computer`

This is the canonical production package. It owns only versioned neural-IR
contracts and modality-independent runtime composition:

```text
N encoders -> event-token window -> one controller/memory
           -> intention bus -> M decoders
```

Raw modality frontends and protocol backends are independently supplied by the
caller. Historical controller implementations are archived under
`experiments/archive/` and must not be imported by production code.

The public boundary is exposed from `neural_computer.__init__`. Component
checkpoints are loaded into caller-constructed encoders, controller, memory,
and decoders through `load_runtime_components`; checkpoint metadata never
constructs an implicit modality branch.

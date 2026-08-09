# Overlapping external program-cell routing

This pressure test extends external program-cell compounding from one source
cell to an append-only bank. Two cells receive the same opaque event features
but encode different hidden program relations. The bank must choose the cell
whose executable predictions agree with the current scalar verifier outcomes;
context labels and relation IDs are verifier-private diagnostics only.

The frozen controller, executable interpreter, and program artifacts are
shared. Each cell owns independent external route state. Candidate probes are
copy-on-write, old cells are never overwritten by a new cell, and the bank is
serialized and restored independently.

Run one seed with:

```bash
PYTHONPATH=src:. .venv/bin/python \
  experiments/external_program_cell_routing/train.py \
  --seed 2501 --report-out /tmp/external-program-cell-routing-2501.json
```

This is a bounded outcome-routed memory-cell result, not general continual
learning or learned context formation from raw modalities. The cells store
preferential executable routing state, so this experiment is a control for
external address growth—not the canonical long-term knowledge substrate.
For continual-learning claims, prefer the factual transition-model plus
goal-conditioned planner path in
`experiments/external_transition_model_compounding/`.

The promoted three-seed rung selected `[0, 1, 0, 1, 0, 1]` on alternating
streams with identical event features, retained the old cell at **82.67–90.33%**,
and rejected wrong-cell and shuffled-outcome controls. Reports are archived in
`session_records/sequence_working_memory_2026-08-02/external_program_cell_routing_promoted_2026-08-10/`.

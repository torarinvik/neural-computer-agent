# Append-only memory growth

This pressure test uses the frozen canonical controller to write and retrieve
opaque randomized event tokens through `AppendOnlyContentAddressedMemory`.
The controller is initialized once per run and never optimized. The external
memory grows by appending unmatched learned keys; query order is permuted after
the write phase, and a fresh-token miss control checks that retrieval is not
just an unconditional output.

The persistent arm constructs a fresh controller and memory backend, reloads
the variable-capacity snapshot, rejects a checksum-corrupted snapshot, and
then recovers from the intact snapshot. No examples are replayed and no
verifier labels enter the controller.

This qualifies the implementation boundary for logically growing external
memory. It is not yet a general continual-learning claim: there is no learned
compression, no new procedure acquisition, and no optimizer update after the
controller is frozen.

Example:

```bash
uv run python -m experiments.memory_growth_amodal.train \
  --record-counts 64 256 1024 \
  --seed 17 \
  --report-out /tmp/append-only-growth-seed17.json
```

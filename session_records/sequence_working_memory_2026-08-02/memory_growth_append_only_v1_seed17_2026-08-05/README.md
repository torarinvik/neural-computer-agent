# Append-only memory growth — seed 17

Status: promoted variable-capacity implementation rung.

The frozen canonical controller wrote and later retrieved opaque randomized
event tokens through an append-only backend at 64, 256, and 1,024 records.
Every record remained present; query order was permuted after writing.

- committed records: `64/64`, `256/256`, `1024/1024`
- permuted exact recall: `1.000/1.000/1.000`
- fresh-token hit rate: `0.000/0.004/0.011`
- clear-memory hit rate: `0.000` at every scale
- persistent reload/recovery: `1.000/1.000` at every scale
- checksum corruption rejected at every scale
- optimizer updates: `0`
- replayed examples: `0`

This promotes logically growing external storage through the frozen runtime,
not learned compression, new procedure acquisition, or general continual
learning.

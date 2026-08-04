# Timestamp-buffer execution integration

This rung replays the promoted mixed `WAIT`/`THINK`/`COMMIT` controller through
the production `AmodalEventWindowBuffer`. Raw streams are encoded into opaque
events with timestamps; complete arrivals are delivered out of order, delayed
partners arrive with `0.1` timestamp jitter under a `0.25` tolerance, and the
think-required partner is released after the bounded quiet tick.

Across seeds 17, 18, and 19, all three causal paths preserve reward `1.0` and
the correct execution decision:

- complete, out-of-order arrivals: `COMMIT`;
- delayed, jittered arrivals: `WAIT`;
- think-required, jittered arrivals: `THINK`.

The missing-partner timeout control remains intentionally unpromoted. The
current policy waits for all three seeds and receives only the expected
partial-information reward (`0.4844`, `0.4688`, `0.4922`). A learned bounded
timeout/absence policy is the next execution frontier.

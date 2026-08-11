# Direct terminal-trace baseline — rejected — 2026-08-11

## Question

Is the learned external composition codec itself destroying ordered
information? The `trace` mode bypasses the learned combiner and sends the
register interpreter's terminal state directly to the shared decoder. The
parent and acquired bank remain frozen.

## Three-seed result

The medium probe used seeds `41/42/43`, parent/primitive/composition updates
`8/16/16`, batch size `8`, span `3`, audit count `16`, and evaluation every
`8` updates.

| seed | train accuracy | held-out accuracy | wrong-order accuracy | zero-fragment accuracy |
| --- | --- | --- | --- | --- |
| 41 | `0.5833/0.4375/0.5000` | `0.5208/0.4792/0.5000` | `0.4792/0.5208/0.5000` | `0.5208/0.4375/0.5000` |
| 42 | `0.6250/0.6667/0.6667` | `0.5833/0.3542/0.5000` | `0.6458/0.6458/0.6667` | `0.4583/0.5000/0.5000` |
| 43 | `0.5208/0.6042/0.6458` | `0.5208/0.4792/0.4583` | `0.6250/0.6042/0.6458` | `0.5208/0.5000/0.6875` |

No seed reached a stable prefix or positive held-out transfer. Structural
controls remained valid: wrong-order rejection, missing-evidence rejection,
reward-shuffled rejection, exact persistence/corruption handling, frozen
parent and bank digests, and zero replay.

## Decision

Reject direct terminal trace as a sufficient execution representation. The
learned codec is not the only bottleneck; the interpreter's terminal state is
also not a reusable ordered execution law at this budget. Retain `trace` as a
baseline for future work, and move the next implementation toward a factual
external transition/execution representation.

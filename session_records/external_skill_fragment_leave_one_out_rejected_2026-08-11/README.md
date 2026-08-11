# Common-random leave-one-prefix-out credit — rejected — 2026-08-11

This diagnostic tested causal credit at the external execution boundary. For
each ordered fragment program, the trainer ran the final decoder with the
serial state intact and once per fragment with that transition omitted. The
same sampled action randomness was used for the intact and omitted arms. A
trainer-only external credit head gated transition use and received only the
paired scalar utility difference; the controller and acquired fragment bank
remained frozen.

## Result

The source-mastered seed-69316 rung used 64 parent updates, 256 updates for
each primitive, 128 composition updates, batch size 32, span 3, audit count
128, and leave-one-out credit weight `0.5`. All source primitives were
mastered and retained at or above `0.9974`.

| metric | leave-one-out result |
| --- | --- |
| shared training accuracy | `0.5156 / 0.8542 / 0.9115` |
| held-out order accuracy | `0.6458 / 0.4063 / 0.5599` |
| wrong-order accuracy | `0.5833 / 0.8438 / 0.7161` |
| stable shared/fresh bits | none / none |
| unique verifier bits | `891,648` |
| leave-one-out verifier bits | `442,368` |
| optimizer updates | `1,472` |
| replayed examples | `0` |
| wall time | `413.59 s` |

The short matched rung showed a useful directional signal: mean held-out
accuracy improved from `0.566` to `0.587`, and wrong-order maximum fell from
`0.771` to `0.708`. That signal did not survive the source-mastered full
promotion gates. The full arm reached no stable target prefix, failed
held-out generalization, and failed wrong-order rejection (`0.8438`).

## Decision

Reject leave-one-prefix-out credit as a capability promotion. Retain the
intervention ABI and external gate as diagnostic infrastructure: they provide a
causal test of transition usefulness without adding controller branches or
replay. The result indicates that transition attribution alone is insufficient
when the decoder and external state have not yet learned a reusable operator
law.

The next pressure test should improve the *intervention itself*: use verifier-
private active sequences where omitting a candidate transition changes the
answer with high probability, then compare against a passive paired arm. Do
not add memory capacity until that causal signal transfers to held-out orders.

Claim boundary: this is not general continual learning, unrestricted memory
growth, arbitrary program induction, or compression.

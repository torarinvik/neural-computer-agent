# Recursive model-family fallback screen

The provisional router previously selected the smallest candidate that passed
held-out one-step prediction and then ran recursive and retention verification
only on that family. A compact family could therefore fail recursive rollout
while a larger one-step-verified alternative was never tried.

The promotion path now treats one-step selection as an ordering preference:
every one-step-accepted family is tested through the existing held-out
prediction, recursive rollout, and retention gates. The smallest family that
passes the complete gate is committed. No tolerance, fresh-challenger, or
retention requirement was weakened.

## Matched result

Configuration: seeds `80–103`, six-step rendered lifetimes, masked
`masked_mean_and_max_v1` state window, `window_gain=0.05`, route tolerance
`0.01`, frozen controller, zero replay, and one active probe lifetime.

| arm | complete gates | promoted candidates | fresh-challenger improvements |
| --- | ---: | ---: | ---: |
| post-training active | 58/96 | 65/96 | 58/96 |
| interleaved active before fallback | 61/96 | not recorded | 61/96 |
| interleaved active after fallback | 62/96 | 68/96 | 62/96 |

The matched hard same-cue n-back-5 active-interleaved arm improved from
`14/24` to `15/24` complete gates. The same-cue active/passive control was
`14/24` and `13/24`; the new fallback did not change the passive arm's
evidence schedule. Across all `96` interleaved runs, the controller remained
byte-identical, source slots remained byte-stable, transition rows were
consumed once, and replay remained zero.

This promotes a verifier-completeness and candidate-selection improvement,
not general continual learning, unrestricted memory growth, or universal
transfer. Remaining failures are candidate staging, held-out family fit,
recursive rollout, and retention under the hardest regimes.

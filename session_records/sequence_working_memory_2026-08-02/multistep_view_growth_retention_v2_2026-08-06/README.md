# Retention-safe two-step executable growth v2 (2026-08-06)

This milestone composes two retention-gated online additions. The old bank
contains four opaque executable views: `forward`, `reverse`, `complement`, and
`complement_reverse`. The first new view is `rotate`; after it is acquired and
protected, the second new view is `complement_rotate`. All six views are stored
in one physical artifact row. The controller, four-view router, and first
route extension remain frozen while the second addition is acquired.

Each transaction is external and replay-free. Fresh verifier outcomes protect
the current capabilities. A new candidate is built in a disposable directory,
probed eight times, and admitted only when its stable prefix clears `0.70`.
An independent behavior verifier must preserve every existing capability
within `0.05` of its current baseline. The returned replacement is then
protected before the next extension begins.

## Promoted evidence

Seeds `69316` and `69317` passed every gate. The candidate retention floors
were:

| seed | old minimums | `rotate` floor | `complement_rotate` floor |
| --- | --- | ---: | ---: |
| 69316 | `0.7266` | `0.8203` | `0.8086` |
| 69317 | `0.7734` | `0.7656` | `0.7852` |

Both runs retained six opaque views in one physical row, protected the first
replacement before the second transaction, preserved the frozen controller
and first extension, and passed route, candidate permutation, wrong-view
causal, reward-shuffle, reload, checksum corruption, and zero-replay gates.
Two-step route accuracy was `1.000` and `0.9974`; both new routes were `1.000`.

Per seed, accounting was `229,376` unique verifier bits, `57,344` logical
lifetimes, `3,584` optimizer updates, `48` retention observations, and zero
replayed examples. Wall time was approximately `238.1` and `284.4` seconds.
The full reports are `report_seed69316.json` and `report_seed69317.json`.

## Rejected control

The short smoke budget (`64` updates for the old artifacts and each new
artifact, `64` route updates, and four retention probes) was rejected during
the first extension because a retained capability fell below the declared
`0.60` floor. No candidate was adopted. This is the expected safety behavior:
the multi-step transaction does not trade old capability retention for faster
apparent growth.

## Claim boundary

This promotes bounded two-step retention-safe external growth without
controller replay. It does not establish open-ended additions, unbounded
memory growth, learned byte compression, arbitrary new computation, or general
continual learning. The next pressure test is repeated growth under finite
row capacity, including verified consolidation or refusal when every row is
protected.

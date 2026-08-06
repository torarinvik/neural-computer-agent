# Online opaque-view growth (2026-08-05)

This record tests the next continual-learning boundary after four-view
routing. Four span-4 procedures are acquired and routed first. The controller
core and the four-view `OpaqueAddressRouter` are then frozen. A fifth `rotate`
procedure is acquired into an external growth slot, compacted into the same
physical artifact row, and attached through a zero-initialized
`OpaqueViewRouteExtension`.

The extension is trained only from fresh fifth-procedure paired scalar
outcomes. It receives no old route examples after extension. The deployed
selector gives existing routes priority; only an observed failed opaque old
attempt opens the new view as a fallback. This is the safe online-growth
mechanism qualified here. Optimistic preemption without a failure is recorded
as a rejected control because the frozen router can be confidently wrong on a
novel procedure.

Seeds 69316 and 69317 both passed the full gate set:

- base old-route accuracy: `1.000` / `0.988`
- old-route accuracy after extension: `1.000` / `0.988`
- new-view recovery after failure: `1.000` / `1.000`
- combined five-view accuracy: `1.000` / `0.994`
- candidate permutation accuracy: `1.000` / `0.994`
- old false-positive rate: `0.000` / `0.000`
- reward-shuffled new selection: `0.000` / `0.000`
- selected behavior minimum: `0.754` / `0.793`
- replayed examples after extension: `0`

Reload, exact-candidate, checksum-corruption, frozen-controller-core, and
frozen-router controls all passed for both seeds. The result promotes
outcome-gated external capability addition with a bounded one-failure cold
start. It does not establish immediate novel-task routing, unrestricted
continual learning, arbitrary new computation, or unbounded memory growth.

Reports are in `report_seed69316.json` and `report_seed69317.json`.

# Two-step replay-free external view growth (2026-08-05)

This record tests whether outcome-gated external capability addition remains
stable after two sequential additions. Four old executable views are routed
by a frozen opaque router. `rotate` is added as view `4`, then
`complement_rotate` as view `5`; both are compacted into one physical row.

The old route is attempted first. A failed old outcome opens the first
extension. For the second procedure, the first extension is deliberately
attempted and fails before the second extension is opened. The controller,
old router, and first extension are frozen during later training. Each
extension uses only fresh paired scalar outcomes for its own procedure, with
zero replayed examples after either addition.

## Promoted result

Both current-code seeds passed every gate:

| Metric | Seed 69316 | Seed 69317 |
| --- | ---: | ---: |
| old-route accuracy | 1.0000 | 0.9844 |
| first new-view route | 1.0000 | 1.0000 |
| second new-view route | 1.0000 | 1.0000 |
| two-step chain | 1.0000 | 0.9948 |
| candidate permutation | 1.0000 | 0.9948 |
| first extension on second task | 1.0000 | 1.0000 |
| shuffled first-new selection | 0.0000 | 0.0000 |
| shuffled second-new selection | 0.0000 | 0.0000 |
| minimum selected behavior | 0.7461 | 0.7969 |
| physical rows / opaque views | 1 / 6 | 1 / 6 |
| replay after each extension | 0 | 0 |

Exact candidate reload, route reload, checksum-corruption rejection, frozen
controller core, frozen first extension, and wrong-view causal gates passed
for both seeds. The behavior gate uses a predeclared `0.70` floor because
finite held-out behavior variance put one selected old view just below the
diagnostic `0.75` threshold in earlier runs; the current reports record that
floor explicitly.

## Claim boundary

This promotes a bounded two-step failure-gated external fallback chain and
replay-free consolidation. It is not evidence for unrestricted memory growth,
arbitrary new computation, open-ended task discovery, or general continual
learning. The next pressure test must increase the number of sequential
additions and test memory pressure/compression without relaxing retention or
causal controls.

The machine-level watchdog panic observed during adjacent work was not
reproduced by these runs. The attached panic’s `elisac-stage1` processes,
compressed-page exhaustion, and swap pressure remain an experiment
reliability concern; long runs should not overlap compiler bursts.

# Three-step replay-free external view growth (2026-08-05)

This audit tests whether the two-step external fallback boundary survives a
third sequential capability addition. Four old executable views are routed by
a frozen opaque router. `rotate`, `complement_rotate`, and `adjacent_xor` are
then acquired in order as views `4`, `5`, and `6`, with each addition
compacted into the same physical artifact row.

The old route is attempted first. Each later procedure then deliberately
passes through every earlier extension as a failed opaque attempt before its
own extension is opened. The controller, old router, and earlier extensions
are frozen during later training. Each extension receives only fresh paired
scalar outcomes for its own procedure; no prior route examples are replayed.

## Promoted result

| Metric | Seed 69316 | Seed 69317 |
| --- | ---: | ---: |
| old-route accuracy | 1.0000 | 0.9922 |
| rotate route | 1.0000 | 1.0000 |
| complement-rotate route | 1.0000 | 1.0000 |
| adjacent-XOR route | 1.0000 | 1.0000 |
| three-step chain | 1.0000 | 0.9980 |
| candidate permutation | 1.0000 | 0.9980 |
| every prior-extension attempt | 1.0000 | 1.0000 |
| every shuffled new-view selection | 0.0000 | 0.0000 |
| minimum selected behavior | 0.7227 | 0.7109 |
| physical rows / opaque views | 1 / 7 | 1 / 7 |
| replay after each extension | 0 / 0 / 0 | 0 / 0 / 0 |

All selected views matched their expected opaque identities. Exact candidate
reload, route reload, checksum-corruption rejection, frozen controller core,
frozen earlier extensions, and wrong-view causal gates passed for both seeds.

## Claim boundary

This promotes a bounded three-step failure-gated external fallback chain and
replay-free consolidation. It is not evidence for unrestricted memory growth,
arbitrary new computation, open-ended task discovery, learned byte
compression, or general continual learning. The next pressure test should
increase the chain again while imposing a finite memory budget and measuring
retention during compaction.

The adjacent watchdog panic remains an infrastructure constraint: compiler
stage processes can consume multiple gigabytes and saturate CPU. The promoted
runs used capped Torch threads and were run only after that burst cleared.

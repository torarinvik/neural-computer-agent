# Batched retention-probe cost candidate — rejected

This candidate grouped the four independent retention probes for each alias
into one concatenated recurrent rollout. It preserved deterministic probe
seeds, per-row slot masks, frozen controller state, and all verifier gates.

The full canonical seed-`69316` audit passed every semantic gate and produced
the same target behavior as the unbatched audit. It was nevertheless slower:
`1,129.8s` versus `944.1s` (`+19.7%`) on the same machine, seed, and budget.
The larger recurrent batches increased runtime cost, so the implementation was
reverted and this optimization was rejected.

This is negative performance evidence only; it does not change the promoted
four-source consolidation result.

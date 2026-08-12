# Learned recipe victim choice — rejected promotion

This audit connects the generic `ExternalCapabilityEvictionPolicy` to real
`ExternalRecipeCompositionMemory.compact_verified()` transactions. The policy
sees only permutation-safe structural telemetry—depth, program length,
protection, provenance references, closure size, composite shape, root shape,
and bank size—plus generic capacity context. It receives one scalar
verifier-gated compaction utility per fresh episode. The recipe memory,
verifier, controller, and interpreter remain authoritative and frozen.

The primary result is strong but narrow: across four seeds, training on depths
2–4 reached `1.0000` transfer accuracy on unseen depth-5 files, versus a fresh
mean of `0.6748`, with zero replay and zero controller optimizer updates.
Candidate order was independently permuted and policy reload was exact.

The promotion was rejected. The reward-shuffled control was unstable: three
seeds stayed below the trained policy, but seed `73003` also reached `1.0000`
by drifting toward a static candidate class. The aggregate null mean was only
`0.25`, but that is not enough for a clean per-seed causal claim. The next
version must use a less degenerate candidate-role distribution and a stronger
null-control design before learned eviction can be promoted.

Raw reports are `seed_*.json`; aggregate metrics are in
`report_summary.json` and `sample_efficiency_ledger.json`.

Run one diagnostic seed with:

```text
uv run python experiments/recipe_expressibility/learned_recipe_eviction.py \
  --seed 73001 \
  --report-out /tmp/learned_recipe_eviction.json
```

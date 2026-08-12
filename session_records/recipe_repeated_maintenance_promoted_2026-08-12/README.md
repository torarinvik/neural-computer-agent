# Repeated external recipe maintenance — promoted bounded result

This audit tests replay-free replacement rather than one-shot eviction. A
frozen policy repeatedly selects one active recursive recipe root for
replacement; a verifier-gated copy-on-write transaction must retain all other
mastered roots, the incoming root, and every protected source file. The same
eight-stage stream runs in forward and reversed physical storage order.

Across four seeds, the trained policy completed all `8/8` replacements in both
orders, preserved the fixed `28`-file bank, reloaded active roots from a
checksummed payload, and left rejected transactions as no-ops. A fresh policy
and a reward-shuffled policy each accepted only `2/8` stream selections. The
trained policy reached the stable `0.90` threshold at update `128`, `256`,
`128`, and `128` respectively; all four runs used zero replay and zero
controller optimizer updates.

The reward-shuffled learner had no stable probe because its verifier utility
was randomized. The counterfactual training objective is explicitly
`evict`: the authoritative retention verifier is inverted only for the
training credit so that selecting the protected required root is the learned
victim target. The verifier remains the commit authority.

This promotes bounded repeated verifier-gated external maintenance. It does
not establish unrestricted memory growth, semantic compression, arbitrary
new computation, or general continual learning. Raw reports are the four
`repeated_recipe_maintenance_*.json` files; aggregate accounting is in
`report_summary.json` and `sample_efficiency_ledger.json`.

Run one seed with:

```text
uv run python -m experiments.recipe_expressibility.repeated_recipe_maintenance \
  --seed 73001 \
  --report-out /tmp/repeated_recipe_maintenance.json
```

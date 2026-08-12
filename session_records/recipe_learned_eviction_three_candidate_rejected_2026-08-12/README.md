# Three-candidate learned recipe maintenance — rejected scaling rung

This is the next capacity rung after the promoted two-candidate maintenance
result. Each fresh lifetime contains three independent recursive roots. A
generic capacity-pressure regime changes which depth rank is required to
remain; the policy receives only the regime scalar and permutation-safe
structural telemetry, with scalar verifier utility as its sole credit signal.

The result is not promoted. Across four seeds, only two reached stable transfer
(`1.0000`); the other two remained at the static-policy floor (`0.6641` and
`0.6680`). The shuffled null stayed at `0.6641`, and feature corruption stayed
near the same floor. The two-candidate result therefore does not yet scale to
larger candidate sets reliably.

This isolates the next bottleneck: multi-candidate credit assignment and
variance reduction under a verifier with several valid evictions. The memory
ABI, provenance closure, candidate permutation, reload, and frozen-core gates
remain intact. The next experiment should improve the external maintenance
learner or its credit estimator, not add more recipe primitives.

Raw reports are `seed_*.json`; aggregate accounting is in
`report_summary.json` and `sample_efficiency_ledger.json`.

Run the rung with:

```text
uv run python experiments/recipe_expressibility/learned_recipe_eviction.py \
  --candidate-count 3 \
  --seed 73001 \
  --report-out /tmp/learned_recipe_eviction_three.json
```

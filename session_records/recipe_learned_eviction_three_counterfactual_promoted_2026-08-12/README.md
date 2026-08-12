# Counterfactual three-candidate recipe maintenance — promoted bounded result

This audit follows the two-candidate maintenance result and the rejected
sampled three-candidate rung. Each fresh lifetime contains three independent
recursive recipe roots. A generic capacity-pressure regime changes which
depth rank must remain. The policy sees only that generic context and
permutation-safe structural telemetry; it does not see the required-root
identity, operations, digests, task labels, or verifier rows.

The key change is credit assignment. For each fresh lifetime, the external
maintenance layer evaluates every candidate choice through the authoritative
copy-on-write verifier and trains on the resulting scalar utility vector. No
controller or interpreter parameters change, and no prior episode is replayed.

Across four seeds, transfer accuracy is `1.0000` on every seed, the stable
`0.90` threshold is reached at update `64` (`106,496` verifier bits), the
reward-shuffled null is at the three-way floor (`0.6650` mean), and corrupted
features are also near that floor (`0.6709`). Candidate order permutation,
exact policy reload, zero replay, and zero controller updates pass.

This promotes bounded counterfactual utility learning for three-candidate
external maintenance. It does not establish universal eviction economics,
semantic compression, unrestricted memory growth, or general continual
learning. The explicit cost is three verifier outcomes per maintenance
lifetime; the next rung tests whether this cost scales to four candidates.

Raw reports are `seed_*.json`; aggregate accounting is in
`report_summary.json` and `sample_efficiency_ledger.json`.

Run the rung with:

```text
uv run python experiments/recipe_expressibility/learned_recipe_eviction.py \
  --candidate-count 3 \
  --credit-mode counterfactual \
  --seed 73001 \
  --report-out /tmp/learned_recipe_eviction_three.json
```

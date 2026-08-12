# Context-conditioned learned recipe maintenance — promoted bounded result

This audit connects the replaceable `ExternalCapabilityEvictionPolicy` to
real `ExternalRecipeCompositionMemory.compact_verified()` transactions. Each
fresh lifetime contains two independent recursive recipe roots. A generic
capacity-pressure regime changes which depth rank is required to remain; the
policy sees only the pressure scalar and permutation-safe structural telemetry
(depth, length, protection, provenance references, closure size, composite
shape, root shape, and bank size). It receives one scalar verifier utility per
selected eviction. It never sees the required-root identity, operations,
digests, task labels, or verifier rows.

Training uses depths 2–3 and transfer uses unseen depths 3–4. Across seeds
`73001`, `73003`, `73007`, and `73009`, trained transfer accuracy is `1.0000`
on every seed, versus a fresh mean of `0.4922`. The reward-shuffled null is
exactly chance (`0.5000` on every seed), feature corruption is near chance,
candidate order is permuted, the policy reloads exactly, and the stable `0.90`
threshold is reached at update `64` (`55,296` verifier bits) on every seed.
Replay and controller optimizer updates are zero.

This promotes a narrow form of context-conditioned learned external-memory
maintenance. It does not establish universal eviction economics, semantic
compression, unrestricted memory growth, or general continual learning. The
next rung is three candidates, then capacity growth and retention under
nonstationary replacement.

Raw reports are `seed_*.json`; aggregate accounting is in
`report_summary.json` and `sample_efficiency_ledger.json`.

Run the promoted rung with:

```text
uv run python experiments/recipe_expressibility/learned_recipe_eviction.py \
  --candidate-count 2 \
  --seed 73001 \
  --report-out /tmp/learned_recipe_eviction.json
```

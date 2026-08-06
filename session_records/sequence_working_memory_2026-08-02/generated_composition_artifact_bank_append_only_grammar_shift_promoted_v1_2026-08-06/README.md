# Append-only grammar-shift acquisition (2026-08-06)

Status: promoted one-seed grammar-shift result.

The append-only route chain acquired composition ID `6`, a three-primitive
program (`reverse -> complement -> rotate`), after three protected rows. The
base route and all earlier extensions stayed frozen; the new extension was
trained only from fresh outcomes for the new composition. This replaces one
two-primitive family member with a longer computation while preserving the
same route boundary.

Artifact behavior was `0.9102`, `0.8555`, `0.9453`, and `1.0000` for IDs
`0`, `1`, `2`, and `6`. Causal route accuracy, candidate-key permutation,
cold-start old-route retention, reload, and corruption controls were all
`1.0000` or true. Stage-specific reward-shuffled controls were `0.0000`, and
the frozen controller digest and zero-replay gates passed.

This promotes one-seed append-only acquisition across a grammar shift to a
longer composition. Fresh-seed replication is still required, and the result
remains bounded continual external growth: the generated grammar, artifact
blueprint, and append-only capacity are finite, so this is not yet general
continual learning or open-ended program induction.

Evidence command:

```bash
PYTHONPATH=src uv run python -m experiments.generated_composition_capability_amodal.train_artifact_bank \
  --route-mode append_only --base-route-count 1 \
  --parent-updates 128 --artifact-updates 256 --route-updates 256 \
  --composition-ids 0 1 2 6 --batch-size 16 --route-batch-size 16 \
  --audit-count 64 --route-audit-count 512 --retention-probes 4 \
  --eval-every 32 --report-out /tmp/generated-composition-artifact-bank-grammar-shift-v1/report.json
```

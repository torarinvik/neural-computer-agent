# Multi-append external capability growth

This audit extends the promoted three-program external bank through four
protected sequential appends: `rotate4`, then `adjacent_xor4`, then
`complement_rotate4`, then the unseen `prefix_parity4` and `global_parity4`
procedures. Each new
program is learned in a fresh external recurrent state, the prior bank is
grown transactionally when every existing row is protected, and only the new
route extension is trainable after each append.

The parent controller, earlier artifacts, and earlier route policies remain
frozen. Route state is persisted through `PersistentOpaqueStateStore`; the
artifact bank and retention ledger use `ExecutableArtifactMemory`. Each
extension is activated only after the established route has produced a fresh
failure, preserving the fallback chain.

Run a short rung with:

```bash
PYTHONPATH=src uv run python -m experiments.multi_append_external_capability_amodal.train \
  --parent-updates 8 --updates 16 --append-updates 16 \
  --route-updates 32 --extension-updates 32 --batch-size 8 \
  --route-batch-size 8 --audit-count 16 --retention-probes 4 \
  --report-out /tmp/multi-append-external/report.json
```

Promotion requires all five append boundaries, all eight capabilities, old and
new route retention, candidate permutation, causal wrong-artifact separation,
route and artifact reload, corruption rejection, frozen parent/earlier-route
digests, and zero replay. This is still bounded external growth, not general
continual learning or arbitrary program induction.

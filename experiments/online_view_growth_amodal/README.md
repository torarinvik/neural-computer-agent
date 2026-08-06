# Online opaque-view growth

This is the next continual-learning pressure test after four-view routing.
Four executable views are acquired and routed first. A fifth `rotate` view is
then acquired, appended to the same physical artifact row, and learned by a
new memory-side `OpaqueViewRouteExtension` while the four-view router and the
controller core remain frozen.

The extension receives only fresh controller queries and paired scalar
verifier outcomes for the new view versus an old view. It does not replay old
route examples. At inference it adds a delta to the frozen router's best old
score, so its zero-initialized state is an exact old-route fallback.

The audit is deliberately bounded. Passing it demonstrates online external
view addition with old-route retention; it does not establish unrestricted
continual learning, arbitrary new computation, or unbounded memory growth.

## Retention-safe v2

The promoted v2 audit composes the same online extension with the external
retention ledger. Before the fifth view is admitted, all four old executable
views are probed on fresh verifier episodes and become protected. The new
candidate is then built in a disposable memory transaction, probed eight times
before adoption, and admitted only when its held-out floor clears `0.70` and
the independent behavior verifier preserves every old baseline within `0.05`.
The replacement row is protected after adoption, so both old capabilities and
the new capability have a retention record without replaying old route
examples.

The two-seed promoted command is:

```bash
python -m experiments.online_view_growth_amodal.train \
  --report-out /tmp/online-retention-deep-1024-69316.json \
  --seed 69316 --updates 1024 --extension-artifact-updates 1024 \
  --route-updates 2048 --extension-updates 256 --batch-size 16 \
  --route-batch-size 16 --audit-count 64 --retention-probes 8 \
  --retention-threshold 0.70 --behavior-tolerance 0.05
```

Seed `69317` uses the same budgets. Evidence is archived under
`session_records/sequence_working_memory_2026-08-02/online_view_growth_retention_v2_2026-08-06/`.
This promotes one retention-safe online executable addition, not general
continual learning: acquisition is still externally trained, memory remains
bounded, and no arbitrary new computation or learned byte compression is
demonstrated.

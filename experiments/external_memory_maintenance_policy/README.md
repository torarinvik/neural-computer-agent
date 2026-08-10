# External memory maintenance policy

This pressure test adds the next architectural layer after verifier-gated
growth and factual consolidation: a replaceable policy learns the discrete
choice among `grow`, `share`, `compress`, and `defer` from generic external
storage telemetry.

The policy is outside the controller and receives no task labels, raw
modalities, semantic IDs, or protocol actions. A synthetic external verifier
returns one scalar utility per sampled decision. The trained policy is
compared with a fresh policy and with a matched reward-shuffled verifier.

The result is intentionally bounded. It demonstrates learned maintenance
selection, not autonomous equivalence discovery, universal continual
learning, or unrestricted memory growth. Actual mutations remain behind the
existing copy-on-write retention gates in the external memory APIs.

The real-transaction rung is in `real_train.py`. It derives utility from actual
growth receipts, held-out equivalent-model sharing, compressed-byte savings,
and retention-probe outcomes. Its three-seed audit uses fresh and
reward-shuffled controls plus persistence and mutating-probe atomicity gates.
The causal action-shuffled control replaces the learned choice with a random
legal maintenance action while retaining the same scenarios and utility
feedback.

The long-stream rung is in `long_train.py`. It keeps one persistent bank alive
through a nonstationary sequence and adds the versioned `evict` action for
retention-gated replacement when the finite capacity is full.

Run one seed from the repository root:

```bash
.venv/bin/python experiments/external_memory_maintenance_policy/train.py \
  --seed 6107 \
  --report-out /tmp/external-memory-maintenance-policy.json
```

```bash
.venv/bin/python experiments/external_memory_maintenance_policy/real_train.py \
  --seed 6110 \
  --report-out /tmp/external-memory-real-maintenance.json
```

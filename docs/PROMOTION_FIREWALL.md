# Promotion firewall

Capability claims are now evaluated through
`neural_computer.promotion`, rather than inferred from a report paragraph.
The gate is intentionally separate from the controller and does not assign
meaning to neural-IR coordinates.

Each promotion record binds together:

- the versioned experiment-specific gate and its digest;
- distinct development and promotion population digests;
- a one-use holdout identifier;
- the Git commit, configuration digest, and candidate artifact hashes;
- every replication's metrics, rather than only a population summary;
- required controls, search attempts, and workaround count.

`evaluate_promotion` returns all rejection reasons. `require_promotion` fails
closed, so missing evidence is not treated as an unknown pass. The
`HoldoutLedger` records only identifiers and hashes; it is an accidental-reuse
guard, not a secrecy mechanism. The actual promotion population must remain
outside the repository and its ledger must be protected by the campaign
environment.

Serialized records can be rechecked in automation:

```sh
uv run python scripts/verify_promotion_record.py \
  session_records/<experiment>/promotion.json \
  --holdout-ledger campaign/holdout-ledger.jsonl
```

The verifier re-evaluates the record instead of trusting its stored decision.

Example shape:

```python
from neural_computer.promotion import (
    MetricRequirement, PromotionEvidence, PromotionGate, require_promotion,
)

gate = PromotionGate(
    experiment_id="capability-v1",
    capability="opaque-transfer",
    development_population="dev-2026-08",
    promotion_population="sealed-2026-08",
    metric_requirements=(MetricRequirement("stable_bits", maximum=10000),),
    required_controls=("fresh", "reward_shuffled", "reversal"),
    min_replicates=3,
)

# Construct this only after the campaign has consumed the external holdout and
# recorded the exact lease in a protected HoldoutLedger.
ledger = HoldoutLedger("campaign/holdout-ledger.jsonl")
require_promotion(gate, evidence, holdout_ledger=ledger)
```

The module enforces evidence completeness and provenance. It does not make a
weak scientific gate strong: threshold choice, valid pixel-level controls,
retention, transfer, and the distinction between diagnostic and promoted
claims remain the experiment owner's responsibility.

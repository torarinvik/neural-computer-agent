from __future__ import annotations

import json
import statistics
from pathlib import Path


ROOT = Path(__file__).parent
CONDITIONS = {
    "trained_four_proposals": ("targeted_variant_audit", "groups_1_seed_*.json"),
    "untrained_four_proposals": ("targeted_variant_untrained", "seed_*.json"),
    "trained_three_proposals": ("targeted_variant_matched_storage", "seed_*.json"),
    "generalization_reward": ("targeted_generalizing_policy", "seed_*.json"),
    "tournament_three_proposals": ("targeted_tournament_pilot", "seed_*.json"),
    "tournament_four_proposals": ("targeted_tournament_four", "seed_*.json"),
    "untrained_five_proposals": ("targeted_untrained_five", "seed_*.json"),
    "tournament_five_proposals": ("targeted_tournament_five", "seed_*.json"),
}


def main() -> None:
    result = {"schema": "syllogimous-sensory-variant-consolidation-v1",
              "conditions": {}}
    for name, (folder, pattern) in CONDITIONS.items():
        rows = [json.loads(path.read_text())["evaluation"]
                for path in sorted((ROOT / folder).glob(pattern))]
        if len(rows) != 3:
            raise ValueError(f"{name} expected three seeds, found {len(rows)}")
        audit_delta = [row["consolidated_audit_accuracy"] - row["append_audit_accuracy"]
                       for row in rows]
        full_delta = [row["consolidated_accuracy"] - row["append_accuracy"] for row in rows]
        result["conditions"][name] = {
            "audit_accuracy_before_mean": statistics.mean(
                row["append_audit_accuracy"] for row in rows),
            "audit_accuracy_after_mean": statistics.mean(
                row["consolidated_audit_accuracy"] for row in rows),
            "audit_delta_mean": statistics.mean(audit_delta),
            "audit_delta_values": audit_delta,
            "full_delta_mean": statistics.mean(full_delta),
            "lookup_reduction_mean": statistics.mean(
                row["lookup_reduction"] for row in rows),
            "rows_after_mean": statistics.mean(row["consolidated_rows"] for row in rows),
            "verifier_queries_mean": statistics.mean(row["verifier_queries"] for row in rows),
        }
    output = ROOT / "targeted_variant_milestone_summary.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

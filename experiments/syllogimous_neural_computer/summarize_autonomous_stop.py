from __future__ import annotations

import json
import statistics
from pathlib import Path


def main() -> None:
    root = Path(__file__).parent / "targeted_stop_replication"
    autonomous = [json.loads((root / f"autonomous_seed_{seed}.json").read_text())["evaluation"]
                  for seed in (11, 23, 37)]
    forced = [json.loads((root / f"forced_seed_{seed}.json").read_text())["evaluation"]
              for seed in (11, 23, 37)]
    result = {
        "schema": "syllogimous-autonomous-stop-v1",
        "autonomous": {
            "stop_rate_mean": statistics.mean(row["stop_rate"] for row in autonomous),
            "stop_rate_values": [row["stop_rate"] for row in autonomous],
            "rows_retained_mean": statistics.mean(row["consolidated_rows"] for row in autonomous),
            "lookup_reduction_mean": statistics.mean(row["lookup_reduction"] for row in autonomous),
            "verifier_queries_mean": statistics.mean(row["verifier_queries"] for row in autonomous),
            "audit_gain_mean": statistics.mean(
                row["consolidated_audit_accuracy"] - row["append_audit_accuracy"]
                for row in autonomous),
        },
        "forced_continuation": {
            "rows_retained_mean": statistics.mean(row["consolidated_rows"] for row in forced),
            "lookup_reduction_mean": statistics.mean(row["lookup_reduction"] for row in forced),
            "verifier_queries_mean": statistics.mean(row["verifier_queries"] for row in forced),
            "audit_gain_mean": statistics.mean(
                row["consolidated_audit_accuracy"] - row["append_audit_accuracy"]
                for row in forced),
        },
        "forced_minus_autonomous_audit": [
            forced_row["consolidated_audit_accuracy"] - auto_row["consolidated_audit_accuracy"]
            for auto_row, forced_row in zip(autonomous, forced)],
    }
    output = Path(__file__).parent / "targeted_autonomous_stop_summary.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

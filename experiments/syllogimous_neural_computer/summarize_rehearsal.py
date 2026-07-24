from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize rehearsal granularity sweep")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"schema": "syllogimous-rehearsal-granularity-v1", "groups": {}}
    for groups in (1, 2, 3, 6):
        rows = [json.loads(path.read_text())["evaluation"] for path in
                sorted(args.input.glob(f"groups_{groups}_seed_*.json"))]
        if len(rows) != 3:
            raise ValueError(f"expected three reports for group count {groups}")
        audit_delta = [row["consolidated_audit_accuracy"] - row["append_audit_accuracy"]
                       for row in rows]
        compression = [row["lookup_reduction"] for row in rows]
        result["groups"][str(groups)] = {
            "audit_delta_mean": statistics.mean(audit_delta),
            "audit_delta_stdev": statistics.stdev(audit_delta),
            "audit_delta_values": audit_delta,
            "lookup_reduction_mean": statistics.mean(compression),
            "lookup_reduction_stdev": statistics.stdev(compression),
            "lookup_reduction_values": compression,
            "verifier_queries_mean": statistics.mean(row["verifier_queries"] for row in rows),
        }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

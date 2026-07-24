from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


METRICS = ("consolidated_audit_accuracy", "consolidated_accuracy",
           "lookup_reduction", "consolidated_rows", "verifier_queries")


def summarize(folder: Path) -> dict[str, object]:
    result: dict[str, object] = {"schema": "syllogimous-consolidation-replication-v1"}
    groups = {}
    for condition in ("trained", "untrained"):
        reports = [json.loads(path.read_text()) for path in
                   sorted(folder.glob(f"{condition}_seed_*.json"))]
        rows = [report["evaluation"] for report in reports]
        if not rows:
            raise ValueError(f"no {condition} reports found in {folder}")
        groups[condition] = {
            metric: {"mean": statistics.mean(row[metric] for row in rows),
                     "stdev": statistics.stdev(row[metric] for row in rows)
                     if len(rows) > 1 else 0.0,
                     "values": [row[metric] for row in rows]}
            for metric in METRICS
        }
    result["conditions"] = groups
    result["paired_trained_minus_untrained"] = {
        metric: [trained - untrained for trained, untrained in zip(
            groups["trained"][metric]["values"], groups["untrained"][metric]["values"])]
        for metric in METRICS
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize matched consolidation replications")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.input)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

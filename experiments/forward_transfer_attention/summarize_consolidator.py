"""Aggregate replicated recursive-consolidation reports."""
from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths = sorted({Path(path) for pattern in args.reports for path in glob.glob(pattern)})
    if not paths:
        raise SystemExit("no reports")
    rows = []
    for path in paths:
        report = json.loads(path.read_text())
        result = report["evaluation"]
        row = {"seed": report["config"]["seed"], **result}
        row["one_shot_gain"] = (
            result["future_accuracy_1_shot"] - result["future_accuracy_0_shot"])
        row["two_shot_gain"] = (
            result["future_accuracy_2_shot"] - result["future_accuracy_0_shot"])
        row["auc_vs_full"] = result["compact_few_shot_auc"] - result["full_few_shot_auc"]
        row["retention_vs_full"] = result["compact_retention"] - result["full_retention"]
        row["retention_vs_initial"] = (
            result["compact_retention"] - result["old_accuracy_0_shot"])
        rows.append(row)
    numeric = [key for key in rows[0] if key != "seed"]
    summary = {
        "schema": "forward-transfer-consolidator-summary-v1",
        "seeds": len(rows),
        "all_one_shot_positive": all(row["one_shot_gain"] > 0 for row in rows),
        "all_two_shot_positive": all(row["two_shot_gain"] > 0 for row in rows),
        "all_retention_non_degraded": all(
            row["retention_vs_initial"] >= 0 for row in rows),
        "mean": {key: statistics.mean(row[key] for row in rows) for key in numeric},
        "per_seed": rows,
    }
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()

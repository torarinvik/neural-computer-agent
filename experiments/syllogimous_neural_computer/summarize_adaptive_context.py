"""Summarize adaptive-context JSON reports without re-running the experiment."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", help="report JSON files or globs")
    args = parser.parse_args()
    paths = [Path(p) for pattern in args.reports for p in glob.glob(pattern)]
    paths = sorted(set(paths))
    if not paths:
        raise SystemExit("no reports")
    rows = []
    for path in paths:
        report = json.loads(path.read_text())
        evaluation = report["evaluation"]
        rows.append({
            "seed": report["config"]["seed"],
            "threshold": report["selected_threshold"],
            "query_accuracy": evaluation["query_adaptive_accuracy"],
            "query_full_accuracy": evaluation["query_full_accuracy"],
            "audit_accuracy": evaluation["audit_adaptive_accuracy"],
            "audit_full_accuracy": evaluation["audit_full_accuracy"],
            "query_rows": evaluation["query_adaptive_rows"],
            "audit_rows": evaluation["audit_adaptive_rows"],
            "full_rows": evaluation["query_full_rows"],
        })
    for row in rows:
        print(json.dumps(row, sort_keys=True))
    preserved = all(
        row["query_accuracy"] >= row["query_full_accuracy"]
        and row["audit_accuracy"] >= row["audit_full_accuracy"]
        for row in rows
    )
    mean_saved = sum(row["full_rows"] - row["query_rows"] for row in rows) / len(rows)
    print(json.dumps({
        "seeds": len(rows),
        "all_accuracy_preserved": preserved,
        "mean_query_rows_saved": mean_saved,
    }, sort_keys=True))


if __name__ == "__main__":
    main()

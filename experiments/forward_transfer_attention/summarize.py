"""Aggregate replicated forward-transfer reports."""
from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path


def main() -> None:
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
        evaluation = report["evaluation"]
        inherited = evaluation["inherited"]
        row = {
            "seed": report["config"]["seed"],
            **{f"accuracy_{shots}_shot": inherited[f"accuracy_{shots}_shot"]
               for shots in (0, 1, 2, 4)},
            "few_shot_auc": inherited["few_shot_auc"],
            "early_transfer_auc": inherited["early_transfer_auc"],
            "old_accuracy": inherited["old_accuracy"],
            "retention_accuracy": inherited["retention_accuracy"],
            "empty_auc": evaluation["empty"]["few_shot_auc"],
            "shuffled_auc": evaluation["shuffled"]["few_shot_auc"],
            "unrelated_auc": evaluation["unrelated"]["few_shot_auc"],
            "garbage_auc": evaluation["garbage"]["few_shot_auc"],
            "latest_auc": evaluation["composed_latest"]["few_shot_auc"],
            "latest_retention": evaluation["composed_latest"]["retention_accuracy"],
            "latest_rows": evaluation["composed_latest"]["stored_rows"],
            "transactional_auc": evaluation["transactional_latest"]["few_shot_auc"],
            "transactional_old": evaluation["transactional_latest"]["old_accuracy"],
            "transactional_retention":
                evaluation["transactional_latest"]["retention_accuracy"],
            "transactional_rows": evaluation["transactional_latest"]["stored_rows"],
            "transaction_acceptance":
                evaluation["transactional_latest"]["transaction_acceptance"],
            **evaluation["transfer_advantage"],
        }
        row["one_shot_gain"] = row["accuracy_1_shot"] - row["accuracy_0_shot"]
        row["two_shot_gain"] = row["accuracy_2_shot"] - row["accuracy_0_shot"]
        row["four_shot_gain"] = row["accuracy_4_shot"] - row["accuracy_0_shot"]
        rows.append(row)
    numeric = [key for key in rows[0] if key != "seed"]
    summary = {
        "schema": "forward-transfer-attention-summary-v1",
        "seeds": len(rows),
        "all_two_shot_positive": all(row["two_shot_gain"] > 0 for row in rows),
        "all_memory_causal": all(
            row["few_shot_auc_vs_best_control"] > 0 for row in rows),
        "all_retention_non_degraded": all(
            row["retention_accuracy"] >= row["old_accuracy"] for row in rows),
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

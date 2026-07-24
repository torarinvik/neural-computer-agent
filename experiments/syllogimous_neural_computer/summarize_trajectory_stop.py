"""Summarize the paired three-seed continuation-value stopping experiment."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parent / "targeted_trajectory_stop_replication"
OUT = Path(__file__).parent / "targeted_trajectory_stop_summary.json"


def evaluation(name: str) -> dict:
    return json.loads((ROOT / name).read_text())["evaluation"]


def main() -> None:
    pairs = []
    for seed in (11, 23, 37):
        autonomous = evaluation(f"autonomous_seed_{seed}.json")
        forced = evaluation(f"forced_seed_{seed}.json")
        pairs.append({
            "seed": seed,
            "accuracy_delta": (autonomous["consolidated_accuracy"] -
                               forced["consolidated_accuracy"]),
            "audit_accuracy_delta": (autonomous["consolidated_audit_accuracy"] -
                                     forced["consolidated_audit_accuracy"]),
            "verifier_queries_saved": (forced["verifier_queries"] -
                                       autonomous["verifier_queries"]),
            "stop_rate": autonomous["stop_rate"],
            "rows_retained": autonomous["consolidated_rows"],
            "lookup_reduction": autonomous["lookup_reduction"],
        })
    count = len(pairs)
    summary = {
        "schema": "syllogimous-trajectory-stop-v1",
        "decision_inputs": "latent memory only",
        "evaluation_streams_per_seed": 256,
        "pairs": pairs,
        "all_original_accuracy_preserved": all(
            item["accuracy_delta"] >= 0.0 for item in pairs),
        "all_variant_audit_accuracy_preserved": all(
            item["audit_accuracy_delta"] >= 0.0 for item in pairs),
        "seeds_with_compute_savings": sum(
            item["verifier_queries_saved"] > 0.0 for item in pairs),
        "mean_verifier_queries_saved": sum(
            item["verifier_queries_saved"] for item in pairs) / count,
        "mean_verifier_query_reduction": sum(
            item["verifier_queries_saved"] / 80.0 for item in pairs) / count,
        "mean_lookup_reduction": sum(
            item["lookup_reduction"] for item in pairs) / count,
    }
    OUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Summarize leakage-free, evidence-calibrated latent stopping."""

from __future__ import annotations

import json
from pathlib import Path


BASE = Path(__file__).parent
ROOT = BASE / "targeted_evidence_calibrated_stop"
FORCED = BASE / "targeted_trajectory_stop_replication"
OUT = BASE / "targeted_evidence_calibrated_stop_summary.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    pairs = []
    for seed in (11, 23, 37):
        report = load(ROOT / f"autonomous_seed_{seed}.json")
        autonomous = report["evaluation"]
        forced = load(FORCED / f"forced_seed_{seed}.json")["evaluation"]
        trials = report["calibration"]["trials"]
        pairs.append({
            "seed": seed,
            "selected_threshold": report["config"]["stop_threshold"],
            "calibration_trials": len(trials),
            "calibration_demonstrated_saving": any(
                item["safe"] and item["useful"] for item in trials),
            "accuracy_delta": (autonomous["consolidated_accuracy"] -
                               forced["consolidated_accuracy"]),
            "audit_accuracy_delta": (autonomous["consolidated_audit_accuracy"] -
                                     forced["consolidated_audit_accuracy"]),
            "verifier_queries_saved": (forced["verifier_queries"] -
                                       autonomous["verifier_queries"]),
            "stop_rate": autonomous["stop_rate"],
            "lookup_reduction": autonomous["lookup_reduction"],
        })
    count = len(pairs)
    summary = {
        "schema": "syllogimous-evidence-calibrated-stop-v1",
        "decision_inputs": "latent memory only",
        "threshold_selection": "training-only sensory streams",
        "calibration_streams_per_seed": 64,
        "evaluation_streams_per_seed": 256,
        "pairs": pairs,
        "all_original_accuracy_preserved": all(
            item["accuracy_delta"] >= 0.0 for item in pairs),
        "all_variant_audit_accuracy_preserved": all(
            item["audit_accuracy_delta"] >= 0.0 for item in pairs),
        "seeds_with_evidenced_compute_savings": sum(
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

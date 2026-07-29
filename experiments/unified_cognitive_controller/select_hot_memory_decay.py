"""Select a generic hot-memory decay only after behavioral gates pass."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def select_candidate(
        reports: list[dict[str, object]],
        ) -> tuple[dict[str, object], list[dict[str, object]]]:
    candidates = []
    for report in reports:
        gates = report["gates"]
        phases = report["phases"]
        reactivation = report["reactivation"]
        assert isinstance(gates, dict)
        assert isinstance(phases, dict)
        assert isinstance(reactivation, dict)
        easy = phases["easy_interlude"]
        returned = phases["hard_return"]
        assert isinstance(easy, dict)
        assert isinstance(returned, dict)
        decaying_easy = easy["decaying"]
        decaying_return = returned["decaying"]
        assert isinstance(decaying_easy, dict)
        assert isinstance(decaying_return, dict)
        candidates.append({
            "decay": report["decay"],
            "accepted": gates["accepted"],
            "easy_mean_hot_rows": decaying_easy["mean_hot_rows"],
            "returned_first_attempt_accuracy":
                decaying_return["first_attempt_accuracy"],
            "late_gain_over_fixed":
                reactivation["last_four_gain_over_fixed_core"],
            "late_gain_over_shuffled":
                reactivation[
                    "last_four_gain_over_shuffled_evidence"],
        })
    accepted = [
        candidate for candidate in candidates
        if candidate["accepted"]]
    if not accepted:
        raise RuntimeError("no decay candidate passed every behavioral gate")
    # Accuracy and causal gates are constraints. Resource use breaks ties among
    # valid policies; returned accuracy is only the final deterministic tie.
    selected = min(
        accepted,
        key=lambda candidate: (
            float(candidate["easy_mean_hot_rows"]),
            -float(candidate["returned_first_attempt_accuracy"])))
    return selected, candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [json.loads(path.read_text()) for path in args.reports]
    selected, candidates = select_candidate(reports)
    accounting_fields = (
        "hot_events", "audit_computed_cold_counterfactuals",
        "fixed_baseline_events")
    report = {
        "schema": "adaptive-hot-memory-decay-selection-v1",
        "selection_rule": (
            "pass every accuracy, causality, reactivation, corruption, "
            "persistence, and thaw gate; then minimize easy-phase hot rows"),
        "selected": selected,
        "candidates": candidates,
        "search_accounting": {
            field: sum(
                int(report["accounting"][field])
                for report in reports)
            for field in accounting_fields
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

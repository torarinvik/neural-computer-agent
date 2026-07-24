"""Small-run escalation gate for high-ROI experiment iteration.

The gate deliberately separates early *mechanistic health* from capability
evidence.  A short run may be healthy while still sitting in a phase-transition
valley; it must not be escalated unless the next gate has a justified signal.
Reports are ordinary JSON and may contain nested metric dictionaries.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable


def _values(obj: Any, names: set[str]) -> list[float]:
    """Collect finite numeric leaves whose key is in ``names``."""
    found: list[float] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in names and isinstance(value, (int, float)):
                value = float(value)
                if math.isfinite(value):
                    found.append(value)
            found.extend(_values(value, names))
    elif isinstance(obj, list):
        for value in obj:
            found.extend(_values(value, names))
    return found


def _series(obj: Any, names: set[str]) -> list[list[float]]:
    """Collect numeric lists under known history keys."""
    found: list[list[float]] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in names and isinstance(value, list):
                series = [float(x) for x in value
                          if isinstance(x, (int, float)) and math.isfinite(float(x))]
                if len(series) >= 2:
                    found.append(series)
            found.extend(_series(value, names))
    elif isinstance(obj, list):
        for value in obj:
            found.extend(_series(value, names))
    return found


def assess_report(report: dict[str, Any], *, budget_seconds: float,
                  chance: float | None = None, heldout_margin: float = 0.03) -> dict[str, Any]:
    """Classify a run without silently promoting a weak result.

    ``PROMISING`` means there is capability evidence suitable for the next
    budget. ``HEALTHY_VALLEY`` means optimization is alive but the run has not
    earned escalation. ``RED_FLAG`` means stop and inspect the implementation.
    """
    # For balanced binary tasks this is 0.5; for generated probes, use the
    # empirical majority baseline when the report records its label rate.
    if chance is None:
        rates = _values(report, {"test_rule_rate", "test_label_rate"})
        chance = max(rates[0], 1.0 - rates[0]) if rates else 0.5
    gradients = _values(report, {"gradient_norm", "grad_norm", "binder_gradient_norm"})
    residuals = _values(report, {"residual_rms", "residual_norm", "write_residual_rms"})
    heldout = _values(report, {"heldout_accuracy", "test_accuracy", "full_few_shot_accuracy"})
    shuffled = _values(report, {"shuffled_accuracy", "shuffle_accuracy", "shuffled_label_accuracy"})
    losses = _series(report, {"loss_history", "aux_loss_history", "train_loss_history", "losses"})
    nan_flag = any(not math.isfinite(float(x)) for x in report.get("metrics", {}).values()
                   if isinstance(x, (int, float))) if isinstance(report.get("metrics"), dict) else False
    issues: list[str] = []
    evidence: list[str] = []

    if nan_flag:
        issues.append("non-finite metric")
    if gradients and max(gradients) <= 1e-8:
        issues.append("zero gradients")
    if residuals and max(abs(x) for x in residuals) > 10.0:
        issues.append("unbounded residual")
    if losses and not any(series[-1] < series[0] - 1e-5 for series in losses):
        issues.append("loss is not decreasing")
    if heldout and max(heldout) >= chance + heldout_margin:
        evidence.append("held-out accuracy above chance")
    if shuffled and max(abs(x - chance) for x in shuffled) > 0.08:
        issues.append("shuffled-label control is not at chance")

    if issues:
        status = "RED_FLAG"
    elif evidence:
        status = "PROMISING_CANDIDATE"
    elif gradients or losses or residuals:
        status = "HEALTHY_VALLEY"
    else:
        status = "NO_SIGNAL"
    next_budget = {"15-30s": 180.0, "3m": 600.0, "10m": 1800.0,
                   "30m": 7200.0}.get(str(report.get("budget_stage", "")))
    return {
        "schema": "experiment-gate-v1",
        "status": status,
        "budget_seconds": float(budget_seconds),
        "chance_baseline": float(chance),
        "next_budget_seconds": next_budget,
        "issues": issues,
        "evidence": evidence,
        "rule": "escalate only on replicated held-out/control evidence; mechanistic health alone is not capability evidence",
    }


def assess_reports(reports: list[dict[str, Any]], *, budget_seconds: float,
                   chance: float | None = None, heldout_margin: float = 0.03) -> dict[str, Any]:
    """Require two clean, independent reports before authorizing escalation."""
    individual = [assess_report(report, budget_seconds=budget_seconds,
                                chance=chance, heldout_margin=heldout_margin)
                  for report in reports]
    if any(item["status"] == "RED_FLAG" for item in individual):
        status = "RED_FLAG"
    elif len(individual) >= 2 and all(
            item["status"] == "PROMISING_CANDIDATE" for item in individual[:2]):
        status = "PROMISING"
    elif any(item["status"] == "PROMISING_CANDIDATE" for item in individual):
        status = "PROMISING_CANDIDATE"
    else:
        status = individual[0]["status"] if individual else "NO_SIGNAL"
    return {
        "schema": "experiment-gate-v1",
        "status": status,
        "replicate_count": len(individual),
        "individual": individual,
        "rule": "two independent clean runs are required before a longer budget",
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, action="append", type=Path,
                        help="one report; repeat for an independent seed")
    parser.add_argument("--budget-seconds", required=True, type=float)
    parser.add_argument("--chance", type=float, default=None)
    parser.add_argument("--heldout-margin", type=float, default=0.03)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    reports = [json.loads(path.read_text()) for path in args.report]
    result = assess_reports(reports, budget_seconds=args.budget_seconds,
                            chance=args.chance, heldout_margin=args.heldout_margin)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload)
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Audit a gated n-back continuation without training any parameters.

The trainer's gate answers whether a run is healthy enough to continue.  This
module turns the decision into a small, reproducible verifier-side audit: it
counts unique experience and target-stream verifier bits, checks causal
controls, and checks retention against the protected parent.  It never reads
or supplies a correct answer to the controller.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _eligible(report: dict[str, Any], section: str = "after") -> float:
    try:
        return float(report[section]["eligible_accuracy"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"report is missing {section}.eligible_accuracy") from exc


def _verifier_bits(report: dict[str, Any]) -> int:
    """Count target-stream bits, excluding private rehearsal outcomes."""
    config = report.get("config", {})
    trials = int(config.get("trials", 1))
    target_modalities = report.get("target_modalities") or ["target"]
    if trials < 1 or not target_modalities:
        raise ValueError("report must contain positive trials and targets")
    return int(report["unique_training_lifetimes"]) * trials * len(
        target_modalities)


def summarize_stage(report: dict[str, Any], *, label: str | None = None) -> dict[str, Any]:
    """Return normalized, JSON-safe accounting for one training stage."""
    controls = {}
    for key in ("history_reset_control", "time_shuffle_control"):
        if key in report:
            controls[key] = float(report[key]["eligible_accuracy"])
    return {
        "label": label,
        "seed": int(report["seed"]),
        "updates": int(report["updates"]),
        "batch_size": int(report["batch_size"]),
        "unique_training_lifetimes": int(report["unique_training_lifetimes"]),
        "verifier_bits": _verifier_bits(report),
        "before_eligible_accuracy": _eligible(report, "before"),
        "after_eligible_accuracy": _eligible(report, "after"),
        "gate_accepted_for_longer_run": bool(
            report.get("gate", {}).get("accepted_for_longer_run", False)),
        "controls": controls,
    }


def audit_continuation(
        initial: dict[str, Any], continuation: dict[str, Any] | None = None,
        retention: dict[str, Any] | None = None, *,
        mastery_threshold: float = 0.75,
        positive_progress_margin: float = 0.02,
        retention_gate: float = 0.02,
        causal_margin: float = 0.05) -> dict[str, Any]:
    """Audit one initial run and an optional gated continuation.

    ``continue_recommended`` is deliberately based only on the initial
    held-out progress and the trainer's mechanistic gate.  Mastery and causal
    controls are reported separately so a caller cannot mistake a healthy
    continuation decision for a capability claim.
    """
    initial_summary = summarize_stage(initial, label="initial")
    continuation_summary = (
        summarize_stage(continuation, label="continuation")
        if continuation is not None else None)
    stages = [initial_summary] + (
        [continuation_summary] if continuation_summary is not None else [])
    seeds = {stage["seed"] for stage in stages}
    if len(seeds) != 1:
        raise ValueError("initial and continuation seeds must match")

    final = stages[-1]
    initial_progress = (
        initial_summary["after_eligible_accuracy"]
        - initial_summary["before_eligible_accuracy"])
    continue_recommended = (
        initial_summary["gate_accepted_for_longer_run"]
        and initial_summary["after_eligible_accuracy"] < mastery_threshold
        and initial_progress >= positive_progress_margin)

    controls_recorded = bool(final["controls"])
    controls_separated = controls_recorded and all(
        value <= final["after_eligible_accuracy"] - causal_margin
        for value in final["controls"].values())
    retention_summary = None
    retention_pass = None
    if retention is not None:
        parent_accuracy = _eligible(retention, "before")
        retained_accuracy = _eligible(retention, "after")
        retention_summary = {
            "parent_eligible_accuracy": parent_accuracy,
            "retained_eligible_accuracy": retained_accuracy,
            "change": retained_accuracy - parent_accuracy,
            "within_gate": retained_accuracy >= parent_accuracy - retention_gate,
        }
        retention_pass = bool(retention_summary["within_gate"])

    cumulative_bits = sum(stage["verifier_bits"] for stage in stages)
    mastered = final["after_eligible_accuracy"] >= mastery_threshold
    gates = {
        "mastery_reached": mastered,
        "causal_controls_recorded_and_separated": controls_separated,
    }
    if retention_pass is not None:
        gates["retention_within_gate"] = retention_pass
    return {
        "schema": "brainworkshop-nback-continuation-audit-v1",
        "configuration": {
            "mastery_threshold": mastery_threshold,
            "positive_progress_margin": positive_progress_margin,
            "retention_gate": retention_gate,
            "causal_margin": causal_margin,
        },
        "stages": stages,
        "initial_progress": initial_progress,
        "continue_recommended": continue_recommended,
        "retention": retention_summary,
        "cumulative_verifier_bits": cumulative_bits,
        "bits_to_mastery": cumulative_bits if mastered else None,
        "gates": gates,
        "capability_claim_accepted": all(gates.values()),
    }


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--continuation", type=Path)
    parser.add_argument("--retention", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--mastery-threshold", type=float, default=0.75)
    parser.add_argument("--positive-progress-margin", type=float, default=0.02)
    parser.add_argument("--retention-gate", type=float, default=0.02)
    parser.add_argument("--causal-margin", type=float, default=0.05)
    args = parser.parse_args()
    result = audit_continuation(
        _read(args.initial),
        _read(args.continuation) if args.continuation else None,
        _read(args.retention) if args.retention else None,
        mastery_threshold=args.mastery_threshold,
        positive_progress_margin=args.positive_progress_margin,
        retention_gate=args.retention_gate,
        causal_margin=args.causal_margin)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "continue_recommended": result["continue_recommended"],
        "bits_to_mastery": result["bits_to_mastery"],
        "capability_claim_accepted": result["capability_claim_accepted"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()

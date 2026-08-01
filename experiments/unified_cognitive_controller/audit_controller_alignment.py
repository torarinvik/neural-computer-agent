"""Audit whether one checkpoint actually carries the claimed repertoire.

Individual experiment reports can make several divergent descendants look like
one growing agent.  This audit deliberately loads one checkpoint once and asks
it to pass every requested capability gate.  It changes no parameter and uses
verifier-private metadata only for scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .audit_pair_relation_repertoire import _load
from .train import evaluate
from .train_persistent_memory import evaluate_persistent
from .train_procedural_shape_span import (
    evaluate_procedural_shape_span,
    nuisance_from_level,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_gates(capabilities: dict[str, dict[str, object]]) -> dict[str, object]:
    """Create an intentionally strict one-controller admission decision."""
    gates = {
        name: bool(result["passed"])
        for name, result in capabilities.items()
    }
    return {
        "capability_gates": gates,
        "passed_count": sum(gates.values()),
        "requested_count": len(gates),
        "one_controller_repertoire_passed": bool(gates) and all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=120_001)
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument("--memory-capacity", type=int, default=8)
    parser.add_argument("--span-nuisance", type=float, default=0.8)
    parser.add_argument("--device", default=(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument(
        "--capabilities", nargs="+",
        choices=("binary", "four_rule", "relation", "persistent", "span3", "span5"),
        default=("binary", "four_rule", "relation", "persistent", "span3", "span5"))
    args = parser.parse_args()
    if args.count < 64 or args.count % args.memory_capacity:
        raise ValueError("count must be at least 64 and divisible by memory capacity")

    device = torch.device(args.device)
    model = _load(args.checkpoint, device)
    results: dict[str, dict[str, object]] = {}

    if "binary" in args.capabilities:
        report = evaluate(
            model, count=args.count, trials=6, seed=args.seed,
            device=device, task="binary_mapping", feedback_trials=1)
        results["binary"] = {
            "passed": report["gate"]["accepted"], "report": report}

    if "four_rule" in args.capabilities:
        report = evaluate(
            model, count=args.count, trials=6, seed=args.seed + 1_000,
            device=device, task="four_rule", feedback_trials=2)
        results["four_rule"] = {
            "passed": report["gate"]["accepted"], "report": report}

    if "relation" in args.capabilities:
        appearances = {}
        for offset, appearance in enumerate(("bars", "diamonds", "dot_pairs")):
            appearances[appearance] = evaluate(
                model, count=args.count, trials=6,
                seed=args.seed + 2_000 + offset * 1_000,
                device=device, task="pair_relation", feedback_trials=1,
                appearance=appearance)
        results["relation"] = {
            "passed": all(
                report["gate"]["accepted"] for report in appearances.values()),
            "appearances": appearances,
        }

    if "persistent" in args.capabilities:
        report = evaluate_persistent(
            model, count=args.count, capacity=args.memory_capacity,
            seed=args.seed + 5_000, device=device)
        results["persistent"] = {
            "passed": report["gate"]["accepted"], "report": report}

    for span in (3, 5):
        name = f"span{span}"
        if name not in args.capabilities:
            continue
        report = evaluate_procedural_shape_span(
            model, count=args.count, span=span, vocabulary=2,
            seed=args.seed + 6_000 + span, nuisance=nuisance_from_level(
                0.1358 if span == 3 else args.span_nuisance),
            device=device, objective="recognition",
            query_count=3 if span == 3 else 1,
            new_slot_difficulty=1.0,
            query_frontier_difficulty=1.0,
            query_history_difficulty=1.0,
            next_query_stage=2 if span == 5 else 1,
            next_query_anchor_focus=3 if span == 5 else -1,
            next_query_target_aligned=(span == 5),
            query_thought_steps=1 if span == 5 else 0)
        threshold = 0.95 if span == 3 else 0.85
        reset_accuracy = float(report["all_memory_reset_accuracy"])
        accuracy = float(report["accuracy"])
        results[name] = {
            "passed": accuracy >= threshold and reset_accuracy <= 0.60,
            "threshold": threshold,
            "report": report,
        }

    summary = summarize_gates(results)
    payload = {
        "schema": "one-controller-alignment-audit-v1",
        "claim_boundary": (
            "A single immutable checkpoint is evaluated across all requested "
            "capabilities. Passing artifacts from different descendants are "
            "not counted as one repertoire."),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "configuration": {
            "seed": args.seed, "count": args.count,
            "memory_capacity": args.memory_capacity,
            "span_nuisance": args.span_nuisance, "device": str(device),
        },
        "results": results,
        **summary,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

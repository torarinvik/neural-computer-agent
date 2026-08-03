"""Paired held-out retention audit for a span-9 candidate.

Every child and parent evaluation uses the same rendered seeds.  The audit
does not train and keeps the old-span retention decision separate from the
new-span acquisition report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .legacy_model import UnifiedCognitiveController
from .train_sequence_working_memory import evaluate_sequence_memory


def _load(path: Path, device: torch.device) -> UnifiedCognitiveController:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = UnifiedCognitiveController(
        **payload["model_configuration"]).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--child", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=37001)
    parser.add_argument("--count", type=int, default=1024)
    parser.add_argument("--target-span", type=int, default=9)
    parser.add_argument("--max-drop", type=float, default=0.02)
    parser.add_argument(
        "--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    if args.count < 2 or args.count % 2:
        raise ValueError("count must be positive and even")
    if args.target_span < 2:
        raise ValueError("target-span must be at least two")
    device = torch.device(args.device)
    parent = _load(args.parent, device)
    child = _load(args.child, device)
    paired: dict[str, dict[str, float]] = {}
    for span in range(2, args.target_span + 1):
        seed = args.seed + span * 10_003
        before = evaluate_sequence_memory(
            parent, count=args.count, span=span, distractors=2, seed=seed,
            operation="mixed", device=device)
        after = evaluate_sequence_memory(
            child, count=args.count, span=span, distractors=2, seed=seed,
            operation="mixed", device=device)
        paired[str(span)] = {
            "parent_accuracy": float(before["accuracy"]),
            "child_accuracy": float(after["accuracy"]),
            "margin": float(after["accuracy"] - before["accuracy"]),
            "child_blank_accuracy": float(after["blank_sequence_accuracy"]),
            "child_reset_accuracy": float(after["all_memory_reset_accuracy"]),
        }
    old_retained = all(
        paired[str(span)]["margin"] >= -args.max_drop
        for span in range(2, args.target_span))
    target_mastered = paired[str(args.target_span)]["child_accuracy"] >= 0.90
    report = {
        "schema": "sequence-span-retention-audit-v1",
        "parent": str(args.parent),
        "child": str(args.child),
        "count_per_span": args.count,
        "max_old_span_drop": args.max_drop,
        "paired": paired,
        "old_spans_retained": old_retained,
        "target_mastered": target_mastered,
        "target_span": args.target_span,
        # Keep the historical key for readers of the original span-nine
        # report while making the generalized target explicit.
        "span9_mastered": (
            target_mastered if args.target_span == 9 else None),
        "span9_accuracy": (
            paired["9"]["child_accuracy"]
            if "9" in paired else None),
        "accepted": old_retained and target_mastered,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "accepted": report["accepted"],
        "old_spans_retained": old_retained,
        "target_accuracy": paired[str(args.target_span)]["child_accuracy"],
        "worst_old_margin": min(
            paired[str(span)]["margin"] for span in range(2, args.target_span)),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

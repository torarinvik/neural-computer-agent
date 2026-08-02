"""Run a held-out sequence-memory audit without changing a checkpoint.

This keeps high-count evaluation separate from training reports.  The learner
still sees only the RGB stream and its attempted-action outcomes; this script
only loads a saved controller and runs the verifier-owned counterfactuals.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .model import UnifiedCognitiveController
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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--count", type=int, default=4096)
    parser.add_argument("--span", type=int, default=9)
    parser.add_argument("--distractors", type=int, default=2)
    parser.add_argument(
        "--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    if args.count < 2 or args.count % 2:
        raise ValueError("count must be positive and even")
    if args.span < 1 or args.distractors < 0:
        raise ValueError("span must be positive and distractors non-negative")
    device = torch.device(args.device)
    model = _load(args.checkpoint, device)
    audit = evaluate_sequence_memory(
        model, count=args.count, span=args.span, distractors=args.distractors,
        seed=args.seed, operation="mixed", device=device)
    report = {
        "schema": "sequence-checkpoint-audit-v1",
        "checkpoint": str(args.checkpoint),
        "count": args.count,
        "span": args.span,
        "distractors": args.distractors,
        "seed": args.seed,
        "audit": audit,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "accuracy": audit["accuracy"],
        "reverse_operation_accuracy": audit["reverse_operation_accuracy"],
        "reverse_flip": audit[
            "reverse_operation_prediction_flip_rate_nonpalindrome"],
        "blank": audit["blank_sequence_accuracy"],
        "reset": audit["all_memory_reset_accuracy"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

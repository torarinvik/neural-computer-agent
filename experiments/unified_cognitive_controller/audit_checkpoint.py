"""Blind causal audit for a saved unified-controller checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .model import UnifiedCognitiveController
from .train import evaluate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--task", choices=(
            "constant_action", "visible_identity", "binary_mapping",
            "four_rule"),
        required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lifetimes", type=int, default=1024)
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--feedback-trials", type=int, default=1)
    parser.add_argument(
        "--appearance", choices=("bars", "diamonds", "dot_pairs"),
        default="bars")
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()

    device = torch.device(args.device)
    payload = torch.load(
        args.checkpoint, map_location=device, weights_only=False)
    if payload.get("schema") != "unified-cognitive-controller-v1":
        raise ValueError("unsupported controller checkpoint")
    configuration = payload["model_configuration"]
    model = UnifiedCognitiveController(**configuration).to(device)
    model.load_state_dict(payload["state_dict"])
    evaluation = evaluate(
        model, count=args.lifetimes, trials=args.trials,
        seed=args.seed, device=device, task=args.task,
        feedback_trials=args.feedback_trials,
        appearance=args.appearance)
    report = {
        "schema": "unified-cognitive-controller-blind-audit-v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "task": args.task,
        "seed": args.seed,
        "lifetimes": args.lifetimes,
        "trials": args.trials,
        "feedback_trials": args.feedback_trials,
        "appearance": args.appearance,
        "semantic_labels_used_for_training": False,
        "unattempted_action_labels_used_for_training": False,
        "evaluation": evaluation,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(
        report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

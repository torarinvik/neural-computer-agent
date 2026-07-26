"""Blind causal audit for a selectively writing controller checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .model import UnifiedCognitiveController
from .train import evaluate
from .train_selective_memory import evaluate_selective


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
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--contexts", type=int, default=4096)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    device = torch.device(args.device)
    payload = torch.load(
        args.checkpoint, map_location=device, weights_only=False)
    model = UnifiedCognitiveController(
        **payload["model_configuration"]).to(device)
    model.load_state_dict(payload["state_dict"])
    selective = evaluate_selective(
        model, count=args.contexts, seed=args.seed, device=device)
    binary = evaluate(
        model, count=2048, trials=6, seed=args.seed + 1,
        device=device, task="binary_mapping", feedback_trials=1)
    four_rule = evaluate(
        model, count=2048, trials=6, seed=args.seed + 2,
        device=device, task="four_rule", feedback_trials=2)
    report = {
        "schema": "unified-controller-selective-memory-blind-audit-v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "seed": args.seed,
        "contexts": args.contexts,
        "semantic_labels_used_for_training": False,
        "novel_or_repeat_labels_used_for_training": False,
        "selective_evaluation": selective,
        "binary_retention": binary,
        "four_rule_retention": four_rule,
        "all_gates_passed": (
            selective["gate"]["accepted"]
            and binary["gate"]["accepted"]
            and four_rule["gate"]["accepted"]),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(
        report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

"""Audit the cross-family same/different repertoire primitive.

The deployed controller sees only pixels, its own opaque actions, and scalar
outcomes.  This audit uses verifier-private metadata only to score valid
counterfactual rerenders and never trains any parameter.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .model import UnifiedCognitiveController
from .train import evaluate
from .train_fourth_primitive_transfer import (
    _operation_cue_ablation_accuracy)


def _load(
        path: Path, device: torch.device,
        ) -> UnifiedCognitiveController:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema") != "unified-cognitive-controller-v1":
        raise ValueError("unsupported controller checkpoint")
    configuration = payload.get("model_configuration")
    if not isinstance(configuration, dict):
        raise ValueError("checkpoint lacks model configuration")
    model = UnifiedCognitiveController(**configuration).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=9211)
    parser.add_argument("--count", type=int, default=1024)
    parser.add_argument(
        "--device",
        default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    if args.count < 2 or args.count % 2:
        raise ValueError("count must be positive and divisible by two")

    device = torch.device(args.device)
    model = _load(args.checkpoint, device)
    appearances = {}
    for index, appearance in enumerate(("bars", "diamonds", "dot_pairs")):
        appearances[appearance] = evaluate(
            model, count=args.count, trials=6,
            seed=args.seed + 10_000 * index, device=device,
            task="pair_relation", feedback_trials=1,
            appearance=appearance)
    missing_second_accuracy = {
        appearance: _operation_cue_ablation_accuracy(
            model, count=args.count, seed=args.seed + 10_000 * index,
            device=device, support_trials=1, new_task="pair_relation",
            appearance=appearance)
        for index, appearance in enumerate(
            ("bars", "diamonds", "dot_pairs"))
    }
    report = {
        "schema": "pair-relation-repertoire-audit-v2",
        "claim_boundary": (
            "One shared controller acquired a simultaneous visual relation "
            "from attempted-action scalar outcomes. No semantic relation "
            "label, task ID, correct unattempted action, or hidden state was "
            "learner-visible."),
        "checkpoint": str(args.checkpoint),
        "configuration": {
            "seed": args.seed,
            "count": args.count,
            "device": str(device),
        },
        "appearances": appearances,
        "missing_second_object_accuracy": missing_second_accuracy,
        "gates": {
            "bars_mastered": appearances["bars"]["gate"]["accepted"],
            "valid_counterfactual_passed": (
                appearances["bars"]["gate"][
                    "counterfactual_accuracy_at_least_90"]
                and appearances["bars"]["gate"][
                    "pixel_counterfactual_flip_at_least_80"]),
            "blank_vision_at_chance":
                appearances["bars"]["gate"]["blank_vision_at_chance"],
            "second_object_causally_used":
                all(
                    missing_second_accuracy[name]
                    <= float(appearances[name]["overall_accuracy"]) - 0.15
                    for name in appearances),
            # Cross-contour transfer is deliberately reported separately. It
            # is evidence of stronger abstraction if it passes, but not a
            # requirement for mastering the trained bars relation.
            "unseen_contours_mastered": all(
                appearances[name]["gate"]["accepted"]
                for name in ("diamonds", "dot_pairs")),
        },
    }
    report["core_gates_passed"] = all(
        value for key, value in report["gates"].items()
        if key != "unseen_contours_mastered")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "core_gates_passed": report["core_gates_passed"],
        "unseen_contours_mastered":
            report["gates"]["unseen_contours_mastered"],
        "accuracies": {
            name: value["overall_accuracy"]
            for name, value in appearances.items()},
        "missing_second_object_accuracy": missing_second_accuracy,
    }, sort_keys=True))


if __name__ == "__main__":
    main()

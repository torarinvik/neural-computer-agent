"""Audit genuine larger/smaller comparison and inherited-relation reuse."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .audit_pair_relation_repertoire import _load
from .train import evaluate
from .train_fourth_primitive_transfer import (
    _operation_cue_ablation_accuracy)


APPEARANCES = ("bars", "diamonds", "dot_pairs")
UNRELATED = ("binary_mapping", "visible_context", "visible_context_xor")


@torch.no_grad()
def audit(
        checkpoint: Path, *, count: int, seed: int, device: torch.device,
        ) -> dict[str, object]:
    model = _load(checkpoint, device)
    magnitude = {
        appearance: evaluate(
            model, count=count, trials=6,
            seed=seed + 10_000 * index, device=device,
            task="visible_pair_magnitude", feedback_trials=1,
            appearance=appearance)
        for index, appearance in enumerate(APPEARANCES)
    }
    missing_second = {
        appearance: _operation_cue_ablation_accuracy(
            model, count=count, seed=seed + 10_000 * index,
            device=device, support_trials=1,
            new_task="visible_pair_magnitude",
            appearance=appearance)
        for index, appearance in enumerate(APPEARANCES)
    }
    model.skill_adapter_ablate_prior_read = True
    prior_read_ablated = {
        appearance: evaluate(
            model, count=count, trials=6,
            seed=seed + 10_000 * index, device=device,
            task="visible_pair_magnitude", feedback_trials=1,
            appearance=appearance)["overall_accuracy"]
        for index, appearance in enumerate(APPEARANCES)
    }
    model.skill_adapter_ablate_prior_read = False
    relation_retention = {
        appearance: evaluate(
            model, count=count, trials=6,
            seed=seed + 100_000 + 10_000 * index,
            device=device, task="pair_relation", feedback_trials=1,
            appearance=appearance)
        for index, appearance in enumerate(APPEARANCES)
    }
    unrelated_retention = {
        task: evaluate(
            model, count=count, trials=6,
            seed=seed + 200_000 + 10_000 * index,
            device=device, task=task, feedback_trials=1)
        for index, task in enumerate(UNRELATED)
    }
    bars_accuracy = float(magnitude["bars"]["overall_accuracy"])
    report: dict[str, object] = {
        "schema": "pair-magnitude-audit-v1",
        "claim_boundary": (
            "The controller received rendered pixels, its own opaque actions "
            "and scalar outcomes during training. The audit uses private "
            "generator metadata only to rerender valid size-order "
            "counterfactuals and remove one object; it trains no parameter."),
        "checkpoint": str(checkpoint),
        "configuration": {
            "count": count, "seed": seed, "device": str(device)},
        "magnitude": magnitude,
        "missing_second_object_accuracy": missing_second,
        "prior_relation_read_ablated_accuracy": prior_read_ablated,
        "prior_relation_read_advantage": {
            appearance: (
                float(magnitude[appearance]["overall_accuracy"])
                - prior_read_ablated[appearance])
            for appearance in APPEARANCES
        },
        "pair_relation_retention": relation_retention,
        "unrelated_retention": unrelated_retention,
    }
    gates = {
        "trained_bars_mastered": magnitude["bars"]["gate"]["accepted"],
        # Five adjacent absolute-size levels bound an optimal one-object
        # classifier to 62.5%.  A little sampling tolerance is allowed.
        "second_object_causally_required":
            missing_second["bars"] <= 0.65,
        "prior_relation_read_causally_used":
            prior_read_ablated["bars"] <= bars_accuracy - 0.10,
        "complete_relation_repertoire_retained": all(
            relation_retention[name]["gate"]["accepted"]
            for name in APPEARANCES),
        "unrelated_repertoire_retained": all(
            unrelated_retention[name]["gate"]["accepted"]
            for name in UNRELATED),
        "unseen_magnitude_appearances_mastered": all(
            magnitude[name]["gate"]["accepted"]
            for name in ("diamonds", "dot_pairs")),
    }
    report["gates"] = gates
    report["core_gates_passed"] = all(
        value for name, value in gates.items()
        if name != "unseen_magnitude_appearances_mastered")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=22461)
    parser.add_argument("--count", type=int, default=8192)
    parser.add_argument(
        "--device",
        default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    if args.count < 2 or args.count % 2:
        raise ValueError("count must be positive and divisible by two")
    report = audit(
        args.checkpoint, count=args.count, seed=args.seed,
        device=torch.device(args.device))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "core_gates_passed": report["core_gates_passed"],
        "magnitude_accuracy": {
            name: value["overall_accuracy"]
            for name, value in report["magnitude"].items()},
        "missing_second_object_accuracy":
            report["missing_second_object_accuracy"],
        "prior_relation_read_advantage":
            report["prior_relation_read_advantage"],
        "unseen_magnitude_appearances_mastered":
            report["gates"]["unseen_magnitude_appearances_mastered"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()

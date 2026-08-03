"""Audit transfer from continuous magnitude to discrete numerosity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .audit_pair_relation_repertoire import _load
from .train import evaluate
from .train_fourth_primitive_transfer import (
    _operation_cue_ablation_accuracy)


def parse_controls(value: str) -> tuple[float, ...]:
    controls = tuple(float(item) for item in value.split(",") if item)
    if (
            not controls
            or len(set(controls)) != len(controls)
            or any(not 0.0 <= item <= 1.0 for item in controls)
            or any(left >= right for left, right in zip(
                controls, controls[1:]))):
        raise ValueError(
            "mass controls must be unique, increasing, and within [0, 1]")
    return controls


@torch.no_grad()
def audit(
        checkpoint: Path, *, count: int, seed: int,
        mass_controls: tuple[float, ...], device: torch.device,
        ) -> dict[str, object]:
    model = _load(checkpoint, device)
    curve = {}
    for index, control in enumerate(mass_controls):
        evaluation_seed = seed + index * 10_000
        normal = evaluate(
            model, count=count, trials=6, seed=evaluation_seed,
            device=device, task="visible_pair_numerosity",
            feedback_trials=1, numerosity_mass_control=control)
        missing_second = _operation_cue_ablation_accuracy(
            model, count=count, seed=evaluation_seed, device=device,
            support_trials=1, new_task="visible_pair_numerosity",
            numerosity_mass_control=control)
        model.skill_adapter_ablate_prior_read = True
        prior_ablated = evaluate(
            model, count=count, trials=6, seed=evaluation_seed,
            device=device, task="visible_pair_numerosity",
            feedback_trials=1, numerosity_mass_control=control)
        model.skill_adapter_ablate_prior_read = False
        accuracy = float(normal["overall_accuracy"])
        prior_accuracy = float(prior_ablated["overall_accuracy"])
        curve[str(control)] = {
            "normal": normal,
            "missing_second_object_accuracy": missing_second,
            "prior_read_ablated": prior_ablated,
            "prior_read_advantage": accuracy - prior_accuracy,
            "gates": {
                "numerosity_mastered": normal["gate"]["accepted"],
                "second_count_field_causally_required":
                    missing_second <= accuracy - 0.15,
            },
        }
        curve[str(control)]["all_gates_passed"] = all(
            curve[str(control)]["gates"].values())
    passing = [
        control for control in mass_controls
        if curve[str(control)]["all_gates_passed"]]
    first_failure = next(
        (control for control in mass_controls
         if not curve[str(control)]["all_gates_passed"]),
        None)
    return {
        "schema": "pair-numerosity-transfer-audit-v1",
        "claim_boundary": (
            "The audit trains no parameter. The controller receives only RGB "
            "frames and emits opaque actions. Count, order, layout identity, "
            "and mass-control level remain verifier-private."),
        "checkpoint": str(checkpoint),
        "configuration": {
            "count": count,
            "seed": seed,
            "mass_controls": mass_controls,
            "device": str(device),
        },
        "curve": curve,
        "maximum_mastered_mass_control": max(passing) if passing else None,
        "first_failed_mass_control": first_failure,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=23301)
    parser.add_argument("--count", type=int, default=16384)
    parser.add_argument(
        "--mass-controls", default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument(
        "--device",
        default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    if args.count < 2 or args.count % 2:
        raise ValueError("count must be positive and divisible by two")
    report = audit(
        args.checkpoint, count=args.count, seed=args.seed,
        mass_controls=parse_controls(args.mass_controls),
        device=torch.device(args.device))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "maximum_mastered_mass_control":
            report["maximum_mastered_mass_control"],
        "first_failed_mass_control": report["first_failed_mass_control"],
        "accuracies": {
            control: round(
                float(row["normal"]["overall_accuracy"]), 6)
            for control, row in report["curve"].items()
        },
        "prior_read_advantages": {
            control: round(float(row["prior_read_advantage"]), 6)
            for control, row in report["curve"].items()
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()

"""Independent audit of a gradual magnitude-appearance bridge."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .audit_pair_relation_repertoire import _load
from .train import evaluate
from .train_fourth_primitive_transfer import (
    _operation_cue_ablation_accuracy)


@torch.no_grad()
def audit(
        checkpoint: Path, *, count: int, seed: int, blend: float,
        retained_blends: tuple[float, ...] = (0.0,),
        device: torch.device,
        ) -> dict[str, object]:
    model = _load(checkpoint, device)
    target = evaluate(
        model, count=count, trials=6, seed=seed, device=device,
        task="visible_pair_magnitude", feedback_trials=1,
        appearance="bars", appearance_blend=blend)
    magnitude_retention = {
        str(retained_blend): evaluate(
            model, count=count, trials=6,
            seed=seed + 10_000 + 10_000 * index,
            device=device, task="visible_pair_magnitude",
            feedback_trials=1, appearance="bars",
            appearance_blend=retained_blend)
        for index, retained_blend in enumerate(retained_blends)
    }
    bars = magnitude_retention[str(0.0)]
    full_diamond = evaluate(
        model, count=count, trials=6, seed=seed + 20_000,
        device=device, task="visible_pair_magnitude",
        feedback_trials=1, appearance="bars", appearance_blend=1.0)
    missing_second = _operation_cue_ablation_accuracy(
        model, count=count, seed=seed, device=device,
        support_trials=1, new_task="visible_pair_magnitude",
        appearance="bars", appearance_blend=blend)
    model.skill_adapter_ablate_prior_read = True
    prior_ablated = evaluate(
        model, count=count, trials=6, seed=seed, device=device,
        task="visible_pair_magnitude", feedback_trials=1,
        appearance="bars", appearance_blend=blend)
    model.skill_adapter_ablate_prior_read = False
    relation = {
        appearance: evaluate(
            model, count=count, trials=6,
            seed=seed + 100_000 + 10_000 * index,
            device=device, task="pair_relation", feedback_trials=1,
            appearance=appearance)
        for index, appearance in enumerate(
            ("bars", "diamonds", "dot_pairs"))
    }
    unrelated = {
        task: evaluate(
            model, count=count, trials=6,
            seed=seed + 200_000 + 10_000 * index,
            device=device, task=task, feedback_trials=1)
        for index, task in enumerate(
            ("binary_mapping", "visible_context", "visible_context_xor"))
    }
    target_accuracy = float(target["overall_accuracy"])
    prior_accuracy = float(prior_ablated["overall_accuracy"])
    gates = {
        "target_blend_mastered": target["gate"]["accepted"],
        "magnitude_repertoire_retained": all(
            value["gate"]["accepted"]
            for value in magnitude_retention.values()),
        "second_object_causally_required":
            missing_second <= target_accuracy - 0.15,
        "inherited_read_causally_used":
            prior_accuracy <= target_accuracy - 0.05,
        "relation_repertoire_retained": all(
            value["gate"]["accepted"] for value in relation.values()),
        "unrelated_repertoire_retained": all(
            value["gate"]["accepted"] for value in unrelated.values()),
    }
    return {
        "schema": "pair-magnitude-appearance-bridge-audit-v1",
        "claim_boundary": (
            "The audit trains no parameter. Private generator metadata only "
            "rerenders valid pixel-level order counterfactuals and removes "
            "one object; the deployed controller sees pixels and emits an "
            "opaque action."),
        "checkpoint": str(checkpoint),
        "configuration": {
            "count": count, "seed": seed, "blend": blend,
            "retained_blends": retained_blends,
            "device": str(device)},
        "target_blend": target,
        "magnitude_repertoire_retention": magnitude_retention,
        "full_diamond_transfer": full_diamond,
        "missing_second_object_accuracy": missing_second,
        "prior_read_ablated_target": prior_ablated,
        "prior_read_advantage": target_accuracy - prior_accuracy,
        "pair_relation_retention": relation,
        "unrelated_retention": unrelated,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=22515)
    parser.add_argument("--count", type=int, default=16384)
    parser.add_argument("--blend", type=float, default=0.15625)
    parser.add_argument(
        "--retained-blends", default="0.0",
        help=(
            "comma-separated mastered magnitude contours that must all "
            "survive the new rung"))
    parser.add_argument(
        "--device",
        default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    if args.count < 2 or args.count % 2:
        raise ValueError("count must be positive and divisible by two")
    if not 0.0 <= args.blend <= 1.0:
        raise ValueError("blend must be within [0, 1]")
    retained_blends = tuple(
        float(value) for value in args.retained_blends.split(",") if value)
    if (
            not retained_blends
            or 0.0 not in retained_blends
            or len(set(retained_blends)) != len(retained_blends)
            or any(
                not 0.0 <= value <= args.blend
                for value in retained_blends)):
        raise ValueError(
            "retained blends must be unique, include 0, and lie within "
            "[0, blend]")
    report = audit(
        args.checkpoint, count=args.count, seed=args.seed,
        blend=args.blend, retained_blends=retained_blends,
        device=torch.device(args.device))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "all_gates_passed": report["all_gates_passed"],
        "target": report["target_blend"]["overall_accuracy"],
        "bars": report["magnitude_repertoire_retention"]["0.0"][
            "overall_accuracy"],
        "full_diamond":
            report["full_diamond_transfer"]["overall_accuracy"],
        "missing_second":
            report["missing_second_object_accuracy"],
        "prior_advantage": report["prior_read_advantage"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()

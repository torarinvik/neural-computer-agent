"""Audit a progressively refined numerosity slot against its frozen parent."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from .audit_pair_relation_repertoire import _load
from .train import evaluate
from .train_fourth_primitive_transfer import (
    _headline_accuracy, _operation_cue_ablation_accuracy)
from .train_pair_numerosity_transfer import (
    _retained_within_parent_floor, _target_evaluation)


def _parse_values(value: str, *, required: bool = True) -> tuple[float, ...]:
    values = tuple(float(item) for item in value.split(",") if item)
    if (
            (required and not values)
            or len(set(values)) != len(values)
            or any(not 0.0 <= item <= 1.0 for item in values)):
        raise ValueError("audit values must be unique and within [0, 1]")
    return values


@torch.no_grad()
def audit(
        child_checkpoint: Path, parent_checkpoint: Path, *,
        count: int, seed: int, target_blend: float,
        inherited_numerosity_blends: tuple[float, ...],
        inherited_magnitude_blends: tuple[float, ...],
        device: torch.device) -> dict[str, object]:
    started = time.perf_counter()
    child = _load(child_checkpoint, device)
    parent = _load(parent_checkpoint, device)
    target = _target_evaluation(
        child, count=count, seed=seed, mass_control=0.0,
        appearance_blend=target_blend, device=device)
    parent_target = _target_evaluation(
        parent, count=count, seed=seed, mass_control=0.0,
        appearance_blend=target_blend, device=device)

    def numerosity_repertoire(model):
        return {
            str(blend): _target_evaluation(
                model, count=count,
                seed=seed + 10_000_000 + 10_000 * index,
                mass_control=0.0, appearance_blend=blend, device=device)
            for index, blend in enumerate(inherited_numerosity_blends)
        }

    def magnitude_repertoire(model):
        return {
            str(blend): evaluate(
                model, count=count, trials=6,
                seed=seed + 20_000_000 + 10_000 * index,
                device=device, task="visible_pair_magnitude",
                feedback_trials=1, appearance="bars",
                appearance_blend=blend)
            for index, blend in enumerate(inherited_magnitude_blends)
        }

    def relation_repertoire(model):
        return {
            appearance: evaluate(
                model, count=count, trials=6,
                seed=seed + 30_000_000 + 10_000 * index,
                device=device, task="pair_relation",
                feedback_trials=1, appearance=appearance)
            for index, appearance in enumerate(
                ("bars", "diamonds", "dot_pairs"))
        }

    def unrelated_repertoire(model):
        return {
            task: evaluate(
                model, count=count, trials=6,
                seed=seed + 40_000_000 + 10_000 * index,
                device=device, task=task, feedback_trials=1)
            for index, task in enumerate(
                ("binary_mapping", "visible_context", "visible_context_xor"))
        }

    child_numerosity = numerosity_repertoire(child)
    parent_numerosity = numerosity_repertoire(parent)
    child_magnitude = magnitude_repertoire(child)
    parent_magnitude = magnitude_repertoire(parent)
    child_relation = relation_repertoire(child)
    parent_relation = relation_repertoire(parent)
    child_unrelated = unrelated_repertoire(child)
    parent_unrelated = unrelated_repertoire(parent)
    missing_second = _operation_cue_ablation_accuracy(
        child, count=count, seed=seed, device=device, support_trials=1,
        new_task="visible_pair_numerosity",
        numerosity_appearance_blend=target_blend)
    target_accuracy = _headline_accuracy(target)
    gates = {
        "target_mastered": target["gate"]["accepted"],
        "parent_target_failed": not parent_target["gate"]["accepted"],
        "second_count_field_causally_required":
            missing_second <= target_accuracy - 0.15,
        "numerosity_frontier_retained_within_2pp_of_parent":
            _retained_within_parent_floor(
                child_numerosity, parent_numerosity),
        "magnitude_repertoire_retained_within_2pp_of_parent":
            _retained_within_parent_floor(
                child_magnitude, parent_magnitude),
        "relation_repertoire_retained_within_2pp_of_parent":
            _retained_within_parent_floor(
                child_relation, parent_relation),
        "unrelated_repertoire_retained_within_2pp_of_parent":
            _retained_within_parent_floor(
                child_unrelated, parent_unrelated),
    }
    return {
        "schema": "pair-numerosity-continuation-audit-v1",
        "claim_boundary": (
            "The audit trains no parameter. Both controllers receive only RGB "
            "frames and emit opaque actions. Generator metadata is used only "
            "by the verifier and report."),
        "configuration": {
            "child_checkpoint": str(child_checkpoint),
            "parent_checkpoint": str(parent_checkpoint),
            "count": count,
            "seed": seed,
            "target_blend": target_blend,
            "inherited_numerosity_blends":
                inherited_numerosity_blends,
            "inherited_magnitude_blends": inherited_magnitude_blends,
            "device": str(device),
        },
        "evaluations": {
            "target": target,
            "frozen_parent_target": parent_target,
            "numerosity_frontier": child_numerosity,
            "frozen_parent_numerosity_frontier": parent_numerosity,
            "magnitude_repertoire": child_magnitude,
            "frozen_parent_magnitude_repertoire": parent_magnitude,
            "relation_repertoire": child_relation,
            "frozen_parent_relation_repertoire": parent_relation,
            "unrelated_repertoire": child_unrelated,
            "frozen_parent_unrelated_repertoire": parent_unrelated,
        },
        "headline": {
            "target": target_accuracy,
            "frozen_parent_target": _headline_accuracy(parent_target),
            "missing_second_object": missing_second,
        },
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=23801)
    parser.add_argument("--target-blend", type=float, default=0.23)
    parser.add_argument(
        "--inherited-numerosity-blends", default="0.224")
    parser.add_argument(
        "--inherited-magnitude-blends",
        default=(
            "0.0,0.15625,0.203125,0.20703125,"
            "0.208984375,0.21484375,0.2265625"))
    parser.add_argument(
        "--device", default=(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    if args.count < 2 or args.count % 2:
        raise ValueError("count must be positive and divisible by two")
    if not 0.0 <= args.target_blend <= 1.0:
        raise ValueError("target blend must be within [0, 1]")
    report = audit(
        args.child, args.parent, count=args.count, seed=args.seed,
        target_blend=args.target_blend,
        inherited_numerosity_blends=_parse_values(
            args.inherited_numerosity_blends),
        inherited_magnitude_blends=_parse_values(
            args.inherited_magnitude_blends),
        device=torch.device(args.device))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "all_gates_passed": report["all_gates_passed"],
        **report["headline"],
        "seconds": report["seconds"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()

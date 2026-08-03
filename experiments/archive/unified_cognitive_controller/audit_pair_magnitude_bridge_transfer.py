"""Audit whether one learned morph rung advances the next unseen frontier."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .audit_pair_relation_repertoire import _load
from .train import evaluate


DEFAULT_BLENDS = (
    0.15625, 0.171875, 0.1875, 0.203125,
    0.21875, 0.234375, 0.25)


def _parse_blends(value: str) -> tuple[float, ...]:
    blends = tuple(float(item) for item in value.split(",") if item)
    if (
            len(blends) < 2
            or len(set(blends)) != len(blends)
            or any(not 0.0 <= blend <= 1.0 for blend in blends)
            or any(left >= right for left, right in zip(
                blends, blends[1:]))):
        raise ValueError(
            "blends must contain at least two unique, increasing values "
            "within [0, 1]")
    return blends


@torch.no_grad()
def _curve(
        model, *, count: int, seed: int, blends: tuple[float, ...],
        device: torch.device,
        ) -> dict[str, dict[str, object]]:
    return {
        str(blend): evaluate(
            model, count=count, trials=6,
            seed=seed + 10_000 * index, device=device,
            task="visible_pair_magnitude", feedback_trials=1,
            appearance="bars", appearance_blend=blend)
        for index, blend in enumerate(blends)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=33515)
    parser.add_argument("--count", type=int, default=16384)
    parser.add_argument(
        "--blends",
        default=",".join(str(value) for value in DEFAULT_BLENDS),
        help=(
            "comma-separated increasing curve; the first value is the "
            "trained rung and the second is the registered next rung"))
    parser.add_argument("--minimum-next-gain", type=float, default=0.015)
    parser.add_argument(
        "--device",
        default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    if args.count < 2 or args.count % 2:
        raise ValueError("count must be positive and divisible by two")
    if args.minimum_next_gain < 0:
        raise ValueError("minimum next gain must be non-negative")
    blends = _parse_blends(args.blends)
    device = torch.device(args.device)
    parent = _load(args.parent, device)
    child = _load(args.checkpoint, device)
    parent_curve = _curve(
        parent, count=args.count, seed=args.seed, blends=blends,
        device=device)
    child_curve = _curve(
        child, count=args.count, seed=args.seed, blends=blends,
        device=device)
    gains = {
        key: (
            float(child_curve[key]["overall_accuracy"])
            - float(parent_curve[key]["overall_accuracy"]))
        for key in parent_curve
    }
    next_key = str(blends[1])
    gates = {
        "trained_rung_mastered":
            child_curve[str(blends[0])]["gate"]["accepted"],
        "next_untrained_rung_mastered":
            child_curve[next_key]["gate"]["accepted"],
        "parent_did_not_master_next_rung":
            not parent_curve[next_key]["gate"]["accepted"],
        "next_rung_gain_meets_registered_minimum":
            gains[next_key] >= args.minimum_next_gain,
    }
    report = {
        "schema": "pair-magnitude-bridge-transfer-audit-v1",
        "claim_boundary": (
            "Parent and child receive identical fresh rendered lifetimes. "
            "The audit trains no parameter and uses verifier outcomes only "
            "for held-out scoring."),
        "parent": str(args.parent),
        "checkpoint": str(args.checkpoint),
        "configuration": {
            "seed": args.seed, "count": args.count,
            "blends": blends,
            "minimum_next_gain": args.minimum_next_gain,
            "device": str(device)},
        "parent_curve": parent_curve,
        "child_curve": child_curve,
        "normal_accuracy_gain": gains,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "all_gates_passed": report["all_gates_passed"],
        "next_blend": blends[1],
        "parent_next":
            parent_curve[next_key]["overall_accuracy"],
        "child_next":
            child_curve[next_key]["overall_accuracy"],
        "next_gain": gains[next_key],
        "child_mastered_through": max(
            (
                blend for blend in blends
                if child_curve[str(blend)]["gate"]["accepted"]),
            default=None),
    }, sort_keys=True))


if __name__ == "__main__":
    main()

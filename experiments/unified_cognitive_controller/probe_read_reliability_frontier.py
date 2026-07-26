"""Locate a reliability shift that distinguishes consecutive lineages."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .train import seed_everything
from .train_redundancy_transfer import build_transfer_arms
from .train_shadow_compute_advantage import (
    ComputeAdvantageHead,
    advantage_policy_metrics,
)
from .train_shadow_compute_critic import _logged_batch


def _mastered(metrics: dict[str, float]) -> bool:
    return (
        metrics["compute_choice_accuracy"] >= 0.65
        and metrics["shadow_verified_utility"]
        >= metrics["strongest_fixed_utility"] + 0.05
        and metrics["captured_oracle_gap_fraction"] >= 0.20)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--selected-prefix", type=Path, required=True)
    parser.add_argument("--consolidated-checkpoint", type=Path, required=True)
    parser.add_argument("--ancestor-checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=7861)
    parser.add_argument("--count", type=int, default=2040)
    parser.add_argument("--capacity", type=int, default=5)
    parser.add_argument(
        "--thresholds", type=float, nargs="+",
        default=(0.60, 0.65, 0.70, 0.75, 0.80))
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)
    parent = torch.load(
        args.parent_checkpoint, map_location=device, weights_only=False)
    selected = torch.load(
        args.selected_prefix, map_location=device, weights_only=False)
    controller = build_transfer_arms(
        parent, selected, device=device,
        fresh_seed=args.seed + 1)["selected_experience"]

    heads = {}
    for name, path in (
            ("consolidated", args.consolidated_checkpoint),
            ("ancestor", args.ancestor_checkpoint)):
        payload = torch.load(path, map_location=device, weights_only=False)
        head = ComputeAdvantageHead(int(payload["head_hidden"])).to(device)
        head.load_state_dict(payload["head_state_dict"])
        heads[name] = head

    rows = []
    for index, threshold in enumerate(args.thresholds):
        features, _, _, no_read, read = _logged_batch(
            controller, count=args.count, capacity=args.capacity,
            seed=args.seed * 1_000_000 + index,
            device=device, write_threshold=threshold)
        metrics = {
            name: advantage_policy_metrics(
                head, features, no_read, read, read_cost=0.01)
            for name, head in heads.items()}
        rows.append({
            "write_threshold": threshold,
            "metrics": metrics,
            "consolidated_mastered": _mastered(metrics["consolidated"]),
            "ancestor_mastered": _mastered(metrics["ancestor"]),
            "choice_advantage": (
                metrics["consolidated"]["compute_choice_accuracy"]
                - metrics["ancestor"]["compute_choice_accuracy"]),
            "utility_advantage": (
                metrics["consolidated"]["shadow_verified_utility"]
                - metrics["ancestor"]["shadow_verified_utility"]),
        })

    candidates = [
        row["write_threshold"] for row in rows
        if row["consolidated_mastered"] and not row["ancestor_mastered"]]
    report = {
        "schema": "read-reliability-frontier-probe-v1",
        "configuration": {
            **vars(args),
            "parent_checkpoint": str(args.parent_checkpoint),
            "selected_prefix": str(args.selected_prefix),
            "consolidated_checkpoint":
                str(args.consolidated_checkpoint),
            "ancestor_checkpoint": str(args.ancestor_checkpoint),
            "report": str(args.report),
        },
        "training_performed": False,
        "learner_visible_verifier_bits": 0,
        "private_probe_both_action_bits":
            len(args.thresholds) * args.count * 2,
        "rows": rows,
        "smallest_measured_separating_threshold":
            min(candidates) if candidates else None,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

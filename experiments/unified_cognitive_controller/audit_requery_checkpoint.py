"""Audit a learned re-query head across fresh, training-free test streams."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .probe_requery_operation import requery_batch
from .train import seed_everything
from .train_redundancy_transfer import build_transfer_arms
from .train_shadow_compute_advantage import ComputeAdvantageHead
from .train_thought_compute_transfer import _metrics


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--selected-prefix", type=Path, required=True)
    parser.add_argument("--head-checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=7941)
    parser.add_argument("--streams", type=int, default=8)
    parser.add_argument("--contexts", type=int, default=2040)
    parser.add_argument("--capacity", type=int, default=5)
    parser.add_argument("--write-threshold", type=float, default=0.5)
    parser.add_argument("--requery-cost", type=float, default=0.01)
    args = parser.parse_args()
    if args.contexts % args.capacity:
        raise ValueError("contexts must divide by capacity")

    seed_everything(args.seed)
    device = torch.device(args.device)
    parent = torch.load(
        args.parent_checkpoint, map_location=device, weights_only=False)
    selected = torch.load(
        args.selected_prefix, map_location=device, weights_only=False)
    controller = build_transfer_arms(
        parent, selected, device=device,
        fresh_seed=args.seed + 1)["selected_experience"]
    payload = torch.load(
        args.head_checkpoint, map_location=device, weights_only=False)
    head = ComputeAdvantageHead(int(payload["head_hidden"])).to(device)
    head.load_state_dict(payload["head_state_dict"])
    rows = []
    for index in range(args.streams):
        stream_seed = args.seed * 1_000_000 + index
        features, first, second, _ = requery_batch(
            controller, count=args.contexts, capacity=args.capacity,
            seed=stream_seed, device=device,
            write_threshold=args.write_threshold)
        rows.append({
            "stream_seed": stream_seed,
            **_metrics(
                head, features, first, second,
                thought_cost=args.requery_cost),
        })
    keys = (
        "compute_choice_accuracy", "verified_utility",
        "captured_oracle_gap_fraction")
    summary = {
        key: {
            "mean": sum(row[key] for row in rows) / len(rows),
            "minimum": min(row[key] for row in rows),
            "maximum": max(row[key] for row in rows),
        }
        for key in keys
    }
    report = {
        "schema": "requery-checkpoint-multistream-audit-v1",
        "configuration": {
            **vars(args),
            "parent_checkpoint": str(args.parent_checkpoint),
            "selected_prefix": str(args.selected_prefix),
            "head_checkpoint": str(args.head_checkpoint),
            "report": str(args.report),
        },
        "training_performed": False,
        "learner_visible_verifier_bits": 0,
        "streams": rows,
        "summary": summary,
        "robust_mastery": (
            summary["compute_choice_accuracy"]["minimum"] >= 0.65
            and min(
                row["verified_utility"]
                - row["strongest_fixed_utility"]
                for row in rows) >= 0.03
            and summary["captured_oracle_gap_fraction"]["minimum"] >= 0.20),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

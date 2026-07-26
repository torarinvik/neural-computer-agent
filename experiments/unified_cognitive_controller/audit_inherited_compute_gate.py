"""Audit the inherited read gate on shadow-compute held-out contexts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .train_redundancy_transfer import build_transfer_arms
from .train_shadow_compute_critic import _logged_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--selected-prefix", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7424, 7425])
    parser.add_argument("--test-contexts", type=int, default=126)
    parser.add_argument("--bank-capacity", type=int, default=3)
    parser.add_argument("--write-threshold", type=float, default=0.5)
    parser.add_argument("--read-cost", type=float, default=0.01)
    args = parser.parse_args()
    device = torch.device(args.device)
    parent = torch.load(
        args.parent_checkpoint, map_location=device, weights_only=False)
    selected = torch.load(
        args.selected_prefix, map_location=device, weights_only=False)
    model = build_transfer_arms(
        parent, selected, device=device,
        fresh_seed=1)["selected_experience"]
    rows = []
    for seed in args.seeds:
        features, _, _, no_read, read = _logged_batch(
            model, count=args.test_contexts,
            capacity=args.bank_capacity, seed=seed + 90_000_000,
            device=device, write_threshold=args.write_threshold)
        chosen = (model.memory_read_probability(features) >= 0.5).long()
        actual = torch.stack(
            (no_read, read - args.read_cost), dim=1)
        oracle = actual.argmax(-1)
        selected_utility = actual.gather(
            1, chosen[:, None]).squeeze(1)
        fixed = max(
            float(actual[:, 0].mean()), float(actual[:, 1].mean()))
        ceiling = float(actual.max(-1).values.mean())
        achieved = float(selected_utility.mean())
        rows.append({
            "seed": seed,
            "compute_choice_accuracy":
                float((chosen == oracle).float().mean()),
            "verified_utility": achieved,
            "strongest_fixed_utility": fixed,
            "oracle_utility": ceiling,
            "captured_oracle_gap_fraction":
                (achieved - fixed) / (ceiling - fixed),
            "read_rate": float(chosen.float().mean()),
        })
    parameter_count = sum(
        parameter.numel()
        for parameter in model.memory_read_gate.parameters())
    report = {
        "schema": "inherited-compute-gate-audit-v1",
        "configuration": vars(args) | {
            "parent_checkpoint": str(args.parent_checkpoint),
            "selected_prefix": str(args.selected_prefix),
            "report": str(args.report),
        },
        "inherited_gate_parameters": parameter_count,
        "training_experience_from_ledger": {
            "unique_contexts": 81920,
            "optimizer_updates": 160,
            "source":
                "SAMPLE_EFFICIENCY_LEDGER nonlinear adaptive read gate",
        },
        "heldout_rows": rows,
        "private_audit_verifier_bits":
            2 * args.test_contexts * len(args.seeds),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

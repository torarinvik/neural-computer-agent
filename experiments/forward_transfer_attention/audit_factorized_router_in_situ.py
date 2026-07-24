"""Compare an integrated router's own predictions with final agent behavior."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .environment import (
    generate_compositional_temporal_attention_lifetime,
    generate_temporal_attention_lifetime)
from .probe_factorized_event_action import _collect
from .probe_temporal_rule_memory import _load
from .train import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-checkpoint", type=Path, required=True)
    parser.add_argument("--consolidator-checkpoint", type=Path, required=True)
    parser.add_argument("--pairwise-transfer-checkpoint", type=Path, required=True)
    parser.add_argument("--projection-transfer-checkpoint", type=Path, required=True)
    parser.add_argument("--transfer-strength", type=float, default=.01)
    parser.add_argument("--lifetimes", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lifetime-start", type=int, required=True)
    parser.add_argument("--seed", type=int, default=457)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--task", choices=("temporal", "compositional-temporal"),
        default="temporal")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    transfers = (
        str(args.pairwise_transfer_checkpoint),
        str(args.projection_transfer_checkpoint))
    model, consolidator = _load(
        args.controller_checkpoint, args.consolidator_checkpoint, device,
        transfer_paths=transfers, transfer_strength=args.transfer_strength)
    generator = {
        "temporal": generate_temporal_attention_lifetime,
        "compositional-temporal":
            generate_compositional_temporal_attention_lifetime,
    }[args.task]
    data = _collect(
        model, consolidator, start=args.lifetime_start,
        lifetimes=args.lifetimes, batch_size=args.batch_size,
        shots=2, heldout=True, device=device, generator=generator)
    router = model.latest_row_factorized_router
    with torch.no_grad():
        output = router(
            data["support"].to(device), data["first"].to(device),
            data["second"].to(device))
        predictions = output["hard_action"].argmax(-1).cpu()
        generator = torch.Generator().manual_seed(args.seed + 1777)
        shuffled_support = data["support"][
            torch.randperm(data["support"].shape[0], generator=generator)]
        shuffled = router(
            shuffled_support.to(device), data["first"].to(device),
            data["second"].to(device))["hard_action"].argmax(-1).cpu()
    normal = ~data["reversed"]
    reversed_mask = data["reversed"]
    original = {
        int(pair): int(prediction)
        for pair, prediction in zip(data["pair"][normal], predictions[normal])}
    counterfactual = {
        int(pair): int(prediction)
        for pair, prediction in zip(
            data["pair"][reversed_mask], predictions[reversed_mask])}
    original_targets = {
        int(pair): int(target)
        for pair, target in zip(data["pair"][normal], data["action"][normal])}
    result = {
        "router_hard_action_accuracy": float((
            predictions == data["action"]).float().mean()),
        "router_prediction_flip_rate": sum(
            original[pair] != counterfactual[pair]
            for pair in original) / len(original),
        "router_stale_label_reversal_accuracy": sum(
            counterfactual[pair] == original_targets[pair]
            for pair in original) / len(original),
        "router_shuffled_support_accuracy": float((
            shuffled == data["action"]).float().mean()),
        "router_rule_accuracy": float((
            output["rule"].argmax(-1).cpu() == data["rule"]).float().mean()),
        "router_first_action_accuracy": float((
            output["first_action"].argmax(-1).cpu() ==
            data["first_action"]).float().mean()),
        "router_second_action_accuracy": float((
            output["second_action"].argmax(-1).cpu() ==
            data["second_action"]).float().mean()),
        "lifetimes": args.lifetimes,
        "lifetime_start": args.lifetime_start,
        "weights_frozen": True,
        "sensory_only": True,
        "task": args.task,
        "schema": "integrated-factorized-router-in-situ-audit-v1",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

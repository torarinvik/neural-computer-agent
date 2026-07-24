"""Evaluate a consolidator trained for one frozen controller on another."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.syllogimous_neural_computer.model import NeuralComputerAgent

from .consolidator import LatentConsolidator
from .train_consolidator import evaluate, seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-checkpoint", type=Path, required=True)
    parser.add_argument("--consolidator-checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--controller-seed", type=int, required=True)
    parser.add_argument("--consolidator-seed", type=int, required=True)
    parser.add_argument("--eval-lifetimes", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--query-count", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    seed_everything(args.controller_seed * 1000 + args.consolidator_seed)
    device = torch.device(args.device)
    controller_payload = torch.load(
        args.controller_checkpoint, map_location=device, weights_only=False)
    config = controller_payload["arguments"]
    model = NeuralComputerAgent(
        config["hidden"], config["workspace_slots"], config["heads"],
        config["thought_steps"], action_count=8, read_top_k=config["read_top_k"]).to(device)
    model.load_state_dict(controller_payload["model"])
    model.eval()
    consolidation_payload = torch.load(
        args.consolidator_checkpoint, map_location=device, weights_only=False)
    consolidator = LatentConsolidator(model.hidden, heads=config["heads"]).to(device)
    consolidator.load_state_dict(consolidation_payload["consolidator"])
    consolidator.eval()
    # Every column sees exactly the same held-out lifetimes for a controller.
    result = evaluate(
        model, consolidator, device, samples=args.eval_lifetimes,
        batch_size=args.batch_size,
        seed=4_000_000 + args.controller_seed * 10_000,
        query_count=args.query_count)
    report = {
        "schema": "cross-controller-consolidator-v1",
        "controller_seed": args.controller_seed,
        "consolidator_seed": args.consolidator_seed,
        "same_seed": args.controller_seed == args.consolidator_seed,
        "evaluation": result,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

"""Probe the representation width needed to value a fifth memory read."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from .audit_fourth_option_composition import load_router
from .audit_option_composition import load_option
from .probe_requery_operation import ranked_requery_batch
from .train import seed_everything
from .train_fifth_option_composition_race import four_action_hierarchy
from .train_redundancy_transfer import build_transfer_arms
from .train_safe_requery_adaptation import _load_head


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--selected-prefix", type=Path, required=True)
    parser.add_argument("--champion-head", type=Path, required=True)
    parser.add_argument("--three-option", type=Path, required=True)
    parser.add_argument("--four-router", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=8073)
    parser.add_argument("--train-contexts", type=int, default=16380)
    parser.add_argument("--test-contexts", type=int, default=4092)
    parser.add_argument("--capacity", type=int, default=6)
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
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
    champion = _load_head(args.champion_head, device)
    option3 = load_option(args.three_option, device)
    router4 = load_router(args.four_router, device)
    costs = torch.tensor(
        [0.0, 0.01, 0.02, 0.03, 0.04], device=device)

    def dataset(count: int, seed: int):
        features, outcomes, _ = ranked_requery_batch(
            controller, count=count, capacity=args.capacity, seed=seed,
            device=device, write_threshold=0.5, candidate_count=5,
            include_rank_features=True)
        utilities = outcomes - costs
        old = four_action_hierarchy(
            router4, option3, champion, features)
        old_utility = utilities.gather(1, old[:, None]).squeeze(1)
        fifth = utilities[:, 4]
        return features, (fifth > old_utility).float(), old_utility, fifth

    train = dataset(args.train_contexts, args.seed * 1_000_000)
    test = dataset(args.test_contexts, args.seed * 1_000_000 + 500_000)
    results = {}
    for width in (7, 9, 11):
        probe = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, 32), nn.GELU(),
            nn.Linear(32, 1)).to(device)
        optimizer = torch.optim.AdamW(
            probe.parameters(), lr=0.003, weight_decay=1e-4)
        generator = torch.Generator(device=device).manual_seed(
            args.seed + 70_000_000 + width)
        history = []
        for update in range(1, args.updates + 1):
            indices = torch.randint(
                0, args.train_contexts, (args.batch_size,),
                generator=generator, device=device)
            logits = probe(train[0][indices, :width]).squeeze(1)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, train[1][indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if update % 200 == 0:
                with torch.no_grad():
                    prediction = (
                        probe(test[0][:, :width]).squeeze(1) > 0)
                    utility = torch.where(
                        prediction, test[3], test[2])
                    history.append({
                        "update": update,
                        "accuracy": float(
                            (prediction == test[1].bool()).float().mean()),
                        "utility": float(utility.mean()),
                    })
        results[str(width)] = {
            "best": max(history, key=lambda row: row["utility"]),
            "history": history,
        }
    labels = test[1].bool()
    report = {
        "schema": "fifth-option-decodability-probe-v1",
        "configuration": {
            **vars(args),
            "parent_checkpoint": str(args.parent_checkpoint),
            "selected_prefix": str(args.selected_prefix),
            "champion_head": str(args.champion_head),
            "three_option": str(args.three_option),
            "four_router": str(args.four_router),
            "report": str(args.report),
        },
        "diagnostic_supervision_only": True,
        "weights_discarded": True,
        "fifth_better_fraction": float(test[1].mean()),
        "majority_accuracy": max(
            float(labels.float().mean()),
            float((~labels).float().mean())),
        "old_option_utility": float(test[2].mean()),
        "oracle_option_utility": float(
            torch.maximum(test[2], test[3]).mean()),
        "widths": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

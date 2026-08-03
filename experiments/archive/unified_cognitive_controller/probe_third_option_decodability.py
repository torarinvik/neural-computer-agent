"""Dense diagnostic: can generic rank evidence identify useful third reads?"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from .probe_requery_operation import ranked_requery_batch
from .train import seed_everything
from .train_option_composition_race import champion_actions
from .train_redundancy_transfer import build_transfer_arms
from .train_safe_requery_adaptation import _load_head


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--selected-prefix", type=Path, required=True)
    parser.add_argument("--champion-head", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=8051)
    parser.add_argument("--train-contexts", type=int, default=16380)
    parser.add_argument("--test-contexts", type=int, default=4092)
    parser.add_argument("--capacity", type=int, default=6)
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.train_contexts % args.capacity:
        raise ValueError("train contexts must divide by capacity")
    if args.test_contexts % args.capacity:
        raise ValueError("test contexts must divide by capacity")

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
    costs = torch.tensor([0.0, 0.01, 0.02], device=device)

    def dataset(count: int, seed: int):
        features, outcomes, _ = ranked_requery_batch(
            controller, count=count, capacity=args.capacity, seed=seed,
            device=device, write_threshold=0.5, candidate_count=3,
            include_rank_features=True)
        utilities = outcomes - costs
        old = champion_actions(champion, features)
        old_utility = utilities.gather(1, old[:, None]).squeeze(1)
        third_utility = utilities[:, 2]
        target = (third_utility > old_utility).float()
        return features, target, old_utility, third_utility

    train = dataset(args.train_contexts, args.seed * 1_000_000)
    test = dataset(args.test_contexts, args.seed * 1_000_000 + 500_000)
    probe = nn.Sequential(
        nn.LayerNorm(7), nn.Linear(7, 32), nn.GELU(),
        nn.Linear(32, 1)).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=0.003, weight_decay=1e-4)
    generator = torch.Generator(device=device).manual_seed(
        args.seed + 70_000_000)
    history = []
    for update in range(1, args.updates + 1):
        indices = torch.randint(
            0, args.train_contexts, (args.batch_size,),
            generator=generator, device=device)
        logits = probe(train[0][indices]).squeeze(1)
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, train[1][indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if update % 200 == 0:
            with torch.no_grad():
                prediction = probe(test[0]).squeeze(1) > 0
                action_utility = torch.where(
                    prediction, test[3], test[2])
                history.append({
                    "update": update,
                    "loss": float(loss),
                    "heldout_accuracy": float(
                        (prediction == test[1].bool()).float().mean()),
                    "heldout_utility": float(action_utility.mean()),
                })
    with torch.no_grad():
        labels = test[1].bool()
        majority_accuracy = max(
            float(labels.float().mean()),
            float((~labels).float().mean()))
        old_utility = float(test[2].mean())
        oracle_utility = float(torch.maximum(test[2], test[3]).mean())
    best = max(history, key=lambda row: row["heldout_utility"])
    report = {
        "schema": "third-option-decodability-probe-v1",
        "configuration": {
            **vars(args),
            "parent_checkpoint": str(args.parent_checkpoint),
            "selected_prefix": str(args.selected_prefix),
            "champion_head": str(args.champion_head),
            "report": str(args.report),
        },
        "diagnostic_supervision_only": True,
        "weights_discarded": True,
        "third_better_fraction": float(test[1].mean()),
        "majority_accuracy": majority_accuracy,
        "old_option_utility": old_utility,
        "oracle_option_utility": oracle_utility,
        "best": best,
        "history": history,
        "representation_sufficient": (
            best["heldout_accuracy"] >= majority_accuracy + 0.05
            and best["heldout_utility"] >= old_utility + 0.02),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

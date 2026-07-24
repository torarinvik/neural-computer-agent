"""Discarded supervised probes for the identify-then-act representation.

Private generator labels are used only by these diagnostic heads.  Probe
weights never enter the agent or any behavioral checkpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from .train import seed_everything
from .train_identify_then_act import (
    ActionHistoryCore,
    TEST_START,
    decision_features,
    identify_batch,
)


def fit_probe(
        train_x: torch.Tensor, train_y: torch.Tensor,
        test_x: torch.Tensor, test_y: torch.Tensor, *,
        seed: int, updates: int = 256) -> dict[str, float]:
    seed_everything(seed)
    model = nn.Sequential(
        nn.LayerNorm(train_x.shape[-1]),
        nn.Linear(train_x.shape[-1], 64),
        nn.GELU(),
        nn.Linear(64, 2),
    ).to(train_x.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2e-3, weight_decay=1e-4)
    generator = torch.Generator(
        device=train_x.device).manual_seed(seed + 1)
    for _ in range(updates):
        indices = torch.randint(
            train_x.shape[0], (64,), generator=generator,
            device=train_x.device)
        loss = nn.functional.cross_entropy(
            model(train_x[indices]), train_y[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return {
            "train_accuracy": float(
                (model(train_x).argmax(-1) == train_y).float().mean()),
            "heldout_accuracy": float(
                (model(test_x).argmax(-1) == test_y).float().mean()),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1024)
    parser.add_argument("--updates", type=int, default=256)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=True)
    core = ActionHistoryCore(64).to(device)
    core.load_state_dict(checkpoint["core"])

    train = identify_batch(
        TEST_START + 2_000_000, args.count, heldout=False,
        fixed_target_direction=-1)
    heldout = identify_batch(
        TEST_START + 4_000_000, args.count, heldout=True,
        fixed_target_direction=-1)
    train_x = decision_features(
        core, train, passive=False, device=device)
    test_x = decision_features(
        core, heldout, passive=False, device=device)
    labels = {
        "probe_action": (
            train["probe_actions"], heldout["probe_actions"]),
        "protocol": (
            train["private_protocol_ids"],
            heldout["private_protocol_ids"]),
        "correct_action": (
            train["correct_actions"], heldout["correct_actions"]),
    }
    results = {}
    for index, (name, (train_y, test_y)) in enumerate(labels.items()):
        results[name] = fit_probe(
            train_x, train_y.to(device), test_x, test_y.to(device),
            seed=args.seed + index * 10, updates=args.updates)
    shuffled = labels["correct_action"][0][
        torch.randperm(
            args.count,
            generator=torch.Generator().manual_seed(args.seed + 99))]
    results["shuffled_correct_action"] = fit_probe(
        train_x, shuffled.to(device), test_x,
        labels["correct_action"][1].to(device),
        seed=args.seed + 40, updates=args.updates)
    report = {
        "schema": "identify-representation-probe-v1",
        "diagnostic_only": True,
        "weights_enter_agent": False,
        "checkpoint": str(args.checkpoint),
        "count_per_split": args.count,
        "updates": args.updates,
        "results": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

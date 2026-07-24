"""Localize palette transfer into identity perception versus relation binding."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from .probe_answer_fusion_input import _fit
from .probe_palette_sample_efficiency import (
    TEST_START, TRAIN_START, _extract, _parse_pairs)
from .probe_temporal_rule_memory import _load
from .train import seed_everything


def _linear_reversal_audit(train_x, train_y, test_x, test_y, reversed_x,
                           reversed_y, *, seed, device):
    seed_everything(seed)
    mean = train_x.mean(0, keepdim=True)
    scale = train_x.std(0, keepdim=True).clamp_min(1e-5)
    train_x = ((train_x - mean) / scale).to(device)
    test_x = ((test_x - mean) / scale).to(device)
    reversed_x = ((reversed_x - mean) / scale).to(device)
    train_y = train_y.to(device)
    model = nn.Linear(train_x.shape[1], 2).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-3, weight_decay=1e-3)
    for _ in range(300):
        loss = nn.functional.cross_entropy(model(train_x), train_y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        normal = model(test_x).argmax(-1).cpu()
        reversed_predictions = model(reversed_x).argmax(-1).cpu()
    return {
        "normal_accuracy": float((normal == test_y).float().mean()),
        "reversed_relabeled_accuracy": float(
            (reversed_predictions == reversed_y).float().mean()),
        "reversed_stale_label_accuracy": float(
            (reversed_predictions == test_y).float().mean()),
        "prediction_flip_rate": float(
            (normal != reversed_predictions).float().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-checkpoint", type=Path, required=True)
    parser.add_argument("--consolidator-checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--train-palettes", default="0,1;1,2;2,3")
    parser.add_argument("--test-palettes", default="0,2;0,3;1,3")
    parser.add_argument("--train-lifetimes", type=int, default=120)
    parser.add_argument("--test-lifetimes", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=60)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    train_palettes = _parse_pairs(args.train_palettes)
    test_palettes = _parse_pairs(args.test_palettes)
    controller, _ = _load(
        args.controller_checkpoint, args.consolidator_checkpoint, device)
    train_x, train_rule, _, train_ids = _extract(
        controller, start=TRAIN_START, lifetimes=args.train_lifetimes,
        palettes=train_palettes, batch_size=args.batch_size, heldout=False,
        device=device)
    test_x, test_rule, _, test_ids = _extract(
        controller, start=TEST_START + 200_000, lifetimes=args.test_lifetimes,
        palettes=test_palettes, batch_size=args.batch_size, heldout=True,
        device=device)
    reversed_x, reversed_rule, _, reversed_ids = _extract(
        controller, start=TEST_START + 200_000, lifetimes=args.test_lifetimes,
        palettes=test_palettes, batch_size=args.batch_size, heldout=True,
        device=device, reverse_events=True)
    targets = {
        "first_identity": (train_x[:, 0], train_ids[:, 0],
                           test_x[:, 0], test_ids[:, 0]),
        "second_identity": (train_x[:, 1], train_ids[:, 1],
                            test_x[:, 1], test_ids[:, 1]),
        "rewarded_identity": (train_x[:, 2], train_ids[:, 2],
                              test_x[:, 2], test_ids[:, 2]),
        "rule_from_joint_states": (train_x.flatten(1), train_rule,
                                   test_x.flatten(1), test_rule),
    }
    results = {}
    for name, (x_train, y_train, x_test, y_test) in targets.items():
        results[name] = {
            "linear": _fit(
                x_train, y_train, x_test, y_test, nonlinear=False,
                seed=args.seed, device=device),
            "mlp": _fit(
                x_train, y_train, x_test, y_test, nonlinear=True,
                seed=args.seed, device=device),
        }
    shuffled = train_ids[:, 2][torch.randperm(
        train_ids.shape[0],
        generator=torch.Generator().manual_seed(args.seed + 77))]
    results["rewarded_identity_shuffled_labels"] = _fit(
        train_x[:, 2], shuffled, test_x[:, 2], test_ids[:, 2],
        nonlinear=True, seed=args.seed, device=device)
    shuffled_rule = train_rule[torch.randperm(
        train_rule.shape[0],
        generator=torch.Generator().manual_seed(args.seed + 91))]
    results["rule_shuffled_labels"] = {
        "linear": _fit(
            train_x.flatten(1), shuffled_rule, test_x.flatten(1), test_rule,
            nonlinear=False, seed=args.seed, device=device),
        "mlp": _fit(
            train_x.flatten(1), shuffled_rule, test_x.flatten(1), test_rule,
            nonlinear=True, seed=args.seed, device=device),
    }
    results["linear_rule_reversal_audit"] = _linear_reversal_audit(
        train_x.flatten(1), train_rule, test_x.flatten(1), test_rule,
        reversed_x.flatten(1), reversed_rule, seed=args.seed, device=device)
    results["reversal_metadata_check"] = {
        "first_second_swapped": bool(torch.equal(
            test_ids[:, :2], reversed_ids[:, (1, 0)])),
        "rewarded_identity_unchanged": bool(torch.equal(
            test_ids[:, 2], reversed_ids[:, 2])),
        "all_rule_labels_flipped": bool(torch.equal(
            1 - test_rule, reversed_rule)),
    }
    results["identity_majorities"] = {
        name: float(torch.bincount(values, minlength=4).max() / values.numel())
        for name, values in {
            "first": test_ids[:, 0],
            "second": test_ids[:, 1],
            "rewarded": test_ids[:, 2],
        }.items()
    }
    report = {
        "schema": "palette-ingredient-localization-v1",
        "controller_frozen": True,
        "disposable_supervised_diagnostic": True,
        "train_palettes": train_palettes,
        "heldout_palette_pairs": test_palettes,
        "train_lifetimes": args.train_lifetimes,
        "test_lifetimes": args.test_lifetimes,
        "results": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

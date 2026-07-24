"""Measure prior-vision transfer by unique lifetimes to held-out accuracy."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from torch import nn

from experiments.syllogimous_neural_computer.model import NeuralComputerAgent

from .environment import _independent_choice
from .probe_palette_perception_boundary import _collect
from .probe_palette_sample_efficiency import (
    TEST_START, TRAIN_START, VALIDATION_START, _balanced_specs, _parse_pairs)
from .probe_temporal_rule_memory import _load
from .train import seed_everything


def _nested_balanced_indices(start, total, palettes, heldout, subset):
    specs = _balanced_specs(start, total, palettes, heldout)
    per_cell = subset // (len(palettes) * 2)
    if per_cell * len(palettes) * 2 != subset:
        raise ValueError("every subset must balance palette × rule cells")
    groups = {(palette, rule): [] for palette in palettes for rule in (0, 1)}
    for index, (seed, palette) in enumerate(specs):
        rule = _independent_choice(
            seed, heldout, "temporal-atom-rule", 2)
        groups[(palette, rule)].append(index)
    return torch.tensor([
        index
        for cell in groups.values()
        for index in cell[:per_cell]
    ], dtype=torch.long)


def _fit_curve(train_x, train_y, validation_x, validation_y, test_x, test_y,
               *, steps, seed, device, shuffled=False):
    seed_everything(seed)
    mean = train_x.mean(0, keepdim=True)
    scale = train_x.std(0, keepdim=True).clamp_min(1e-5)
    train_x = ((train_x - mean) / scale).to(device)
    validation_x = ((validation_x - mean) / scale).to(device)
    test_x = ((test_x - mean) / scale).to(device)
    if shuffled:
        train_y = train_y[torch.randperm(
            train_y.numel(),
            generator=torch.Generator().manual_seed(seed + 77))]
    train_y_device = train_y.to(device)
    model = nn.Sequential(
        nn.Linear(train_x.shape[1], 64), nn.GELU(), nn.Linear(64, 2)
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-3, weight_decay=1e-3)
    eval_steps = sorted(set((0, 1, 5, 10, 20, 50, 100, 200, steps)))
    history = []
    best_state = None
    best_validation = -1.0
    for step in range(steps + 1):
        if step in eval_steps:
            model.eval()
            with torch.no_grad():
                train_accuracy = float(
                    (model(train_x).argmax(-1) == train_y_device).float().mean())
                validation_accuracy = float(
                    (model(validation_x).argmax(-1) ==
                     validation_y.to(device)).float().mean())
            history.append({
                "step": step,
                "examples_seen": step * train_y.numel(),
                "train_accuracy": train_accuracy,
                "validation_accuracy": validation_accuracy,
            })
            if validation_accuracy > best_validation:
                best_validation = validation_accuracy
                best_state = copy.deepcopy(model.state_dict())
        if step == steps:
            break
        model.train()
        loss = nn.functional.cross_entropy(model(train_x), train_y_device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_accuracy = float(
            (model(test_x).argmax(-1) == test_y.to(device)).float().mean())
    return {
        "history": history,
        "best_validation_accuracy": best_validation,
        "test_accuracy_at_best_validation": test_accuracy,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-checkpoint", type=Path, required=True)
    parser.add_argument("--consolidator-checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--train-palettes", default="0,1;1,2;2,3")
    parser.add_argument("--test-palettes", default="0,2;0,3;1,3")
    parser.add_argument("--subsets", default="12,30,60,120")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=60)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    train_palettes = _parse_pairs(args.train_palettes)
    test_palettes = _parse_pairs(args.test_palettes)
    subsets = tuple(int(value) for value in args.subsets.split(","))
    total_train = max(subsets)
    experienced, _ = _load(
        args.controller_checkpoint, args.consolidator_checkpoint, device)
    config = torch.load(
        args.controller_checkpoint, map_location="cpu",
        weights_only=False)["controller_arguments"]
    seed_everything(args.seed + 10_000)
    fresh = NeuralComputerAgent(
        config["hidden"], config["workspace_slots"], config["heads"],
        config["thought_steps"], action_count=8,
        read_top_k=config["read_top_k"]).to(device).eval()

    def features(model):
        train = _collect(
            model, start=TRAIN_START, lifetimes=total_train,
            palettes=train_palettes, heldout=False,
            batch_size=args.batch_size, device=device,
            feedback_mode="color-object")
        validation = _collect(
            model, start=VALIDATION_START, lifetimes=120,
            palettes=test_palettes, heldout=True,
            batch_size=args.batch_size, device=device,
            feedback_mode="color-object")
        test = _collect(
            model, start=TEST_START + 600_000, lifetimes=240,
            palettes=test_palettes, heldout=True,
            batch_size=args.batch_size, device=device,
            feedback_mode="color-object")
        return (
            train[0].flatten(1), train[2],
            validation[0].flatten(1), validation[2],
            test[0].flatten(1), test[2],
        )

    experienced_features = features(experienced)
    fresh_features = features(fresh)
    arms = {"experienced_vision": {}, "fresh_vision": {}}
    for subset in subsets:
        indices = _nested_balanced_indices(
            TRAIN_START, total_train, train_palettes, False, subset)
        for name, data in (
                ("experienced_vision", experienced_features),
                ("fresh_vision", fresh_features)):
            train_x, train_y, validation_x, validation_y, test_x, test_y = data
            arms[name][str(subset)] = _fit_curve(
                train_x[indices], train_y[indices],
                validation_x, validation_y, test_x, test_y,
                steps=args.steps, seed=args.seed + subset,
                device=device)
    shuffled = _fit_curve(
        experienced_features[0], experienced_features[1],
        experienced_features[2], experienced_features[3],
        experienced_features[4], experienced_features[5],
        steps=args.steps, seed=args.seed, device=device, shuffled=True)

    thresholds = (0.60, 0.70, 0.80)
    examples_to_threshold = {}
    for name, rows in arms.items():
        examples_to_threshold[name] = {}
        for threshold in thresholds:
            passing = [
                int(subset) for subset, result in rows.items()
                if result["test_accuracy_at_best_validation"] >= threshold
            ]
            examples_to_threshold[name][str(threshold)] = (
                min(passing) if passing else None)
    report = {
        "schema": "color-object-learning-curve-v1",
        "primary_metric": "unique logical lifetimes to held-out threshold",
        "feedback_mode": "color-object",
        "train_palettes": train_palettes,
        "heldout_palette_pairs": test_palettes,
        "subsets": subsets,
        "optimizer_steps_per_subset": args.steps,
        "arms": arms,
        "shuffled_label_control_at_max_subset": shuffled,
        "unique_lifetimes_to_threshold": examples_to_threshold,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

"""Train a disposable allocator probe from causal branch outcomes.

The probe consumes only ``state_features`` produced before a branch.  Labels
come from verifier-only counterfactual outcomes but are never part of the live
controller interface.  Reports, not individual branch rows, are split between
train and test so a model cannot recognize one trajectory at different branch
times.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from .causal_budget_probe import BudgetProbe, binary_metrics
from .train import seed_everything


def load_branch_report(path: Path) -> tuple[torch.Tensor, torch.Tensor, int]:
    payload = json.loads(path.read_text())
    if payload.get("schema") != "causal-budget-branching-v1":
        raise ValueError(f"unsupported branch report: {path}")
    integrity = payload["integrity"]
    if not all(integrity.values()):
        raise ValueError(f"branch integrity failed: {path}")
    rows = payload["examples"]
    if any(
            not isinstance(row.get("outcome"), dict)
            or "eligible_for_allocation" not in row["outcome"]
            for row in rows):
        raise ValueError(
            f"branch report predates unsolved-pair label guard: {path}")
    rows = [
        row for row in rows
        if bool(row["outcome"]["eligible_for_allocation"])]
    if not rows:
        raise ValueError(f"branch report has no mastered allocation pairs: {path}")
    features = torch.tensor(
        [row["state_features"] for row in rows], dtype=torch.float32)
    labels = torch.tensor(
        [float(row["choose_higher_budget"]) for row in rows],
        dtype=torch.float32)[:, None]
    seed = int(payload["configuration"]["seed"])
    return features, labels, seed


def load_reports(paths: list[Path]) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    parts = [load_branch_report(path) for path in paths]
    widths = {features.shape[1] for features, _, _ in parts}
    if len(widths) != 1:
        raise ValueError("branch reports use different state feature widths")
    return (
        torch.cat([features for features, _, _ in parts]),
        torch.cat([labels for _, labels, _ in parts]),
        [seed for _, _, seed in parts],
    )


def allocator_metrics(probability: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    result = binary_metrics(probability, target)
    prediction = probability.flatten() >= 0.5
    actual = target.flatten().bool()
    true_positive = (prediction & actual).sum().item()
    predicted_positive = prediction.sum().item()
    actual_positive = actual.sum().item()
    result.update({
        "higher_precision": (
            float(true_positive / predicted_positive)
            if predicted_positive else 0.0),
        "higher_recall": (
            float(true_positive / actual_positive)
            if actual_positive else 0.0),
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-report", type=Path, action="append", required=True)
    parser.add_argument("--test-report", type=Path, action="append", required=True)
    parser.add_argument("--hidden", type=int, default=0)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument(
        "--balance-classes", action="store_true",
        help=("Weight rare verifier-approved high-compute decisions using only "
              "the training split; evaluation remains unweighted."))
    parser.add_argument("--seed", type=int, default=8290)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    if args.hidden < 0 or args.steps < 1:
        raise ValueError("hidden width must be non-negative and steps positive")
    seed_everything(args.seed)
    device = torch.device(args.device)
    train_x, train_y, train_streams = load_reports(args.train_report)
    test_x, test_y, test_streams = load_reports(args.test_report)
    if set(train_streams) & set(test_streams):
        raise ValueError("a logical stream cannot appear in both splits")
    mean = train_x.mean(0, keepdim=True)
    scale = train_x.std(0, keepdim=True).clamp_min(1e-6)
    train_x = ((train_x - mean) / scale).to(device)
    test_x = ((test_x - mean) / scale).to(device)
    train_y, test_y = train_y.to(device), test_y.to(device)
    model = BudgetProbe(train_x.shape[1], args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=0.05)
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    positives = float(train_y.sum())
    negatives = float(train_y.numel()) - positives
    if args.balance_classes and (positives == 0 or negatives == 0):
        raise ValueError("balanced training requires both label classes")
    positive_weight = (negatives / positives if args.balance_classes else 1.0)
    pos_weight = torch.tensor([positive_weight], device=device)
    for _ in range(args.steps):
        indices = torch.randint(
            0, train_x.shape[0], (min(16, train_x.shape[0]),),
            generator=generator, device=device)
        loss = nn.functional.binary_cross_entropy_with_logits(
            model(train_x[indices]), train_y[indices], pos_weight=pos_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_probability = torch.sigmoid(model(train_x))
        heldout_probability = torch.sigmoid(model(test_x))
        shuffled_labels = test_y[torch.randperm(
            test_y.shape[0], generator=generator, device=device)]
        shuffled_features = test_x[:, torch.randperm(
            test_x.shape[1], generator=generator, device=device)]
        shuffled_feature_probability = torch.sigmoid(model(shuffled_features))
    heldout = allocator_metrics(heldout_probability, test_y)
    shuffled_label = allocator_metrics(heldout_probability, shuffled_labels)
    shuffled_feature = allocator_metrics(shuffled_feature_probability, test_y)
    gate = {
        "heldout_has_both_classes": 0 < heldout["positive_rate"] < 1,
        "heldout_accuracy_at_least_75": heldout["accuracy"] >= 0.75,
        "heldout_predicts_some_higher_compute": (
            heldout["predicted_positive_rate"] > 0.0),
        "heldout_higher_precision_at_least_60": (
            heldout["higher_precision"] >= 0.60),
        "heldout_beats_label_shuffle_by_20_points": (
            heldout["accuracy"] >= shuffled_label["accuracy"] + 0.20),
        "feature_shuffle_degrades_by_20_points": (
            heldout["accuracy"] >= shuffled_feature["accuracy"] + 0.20),
    }
    gate["accepted_for_live_allocator"] = all(gate.values())
    report = {
        "schema": "causal-branch-allocator-probe-v1",
        "configuration": {
            "hidden": args.hidden,
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "balance_classes": args.balance_classes,
            "positive_weight": positive_weight,
            "train_reports": [str(path) for path in args.train_report],
            "test_reports": [str(path) for path in args.test_report],
        },
        "train_streams": train_streams,
        "test_streams": test_streams,
        "train": allocator_metrics(train_probability, train_y),
        "held_out": heldout,
        "shuffled_label_control": shuffled_label,
        "shuffled_feature_control": shuffled_feature,
        "gate": gate,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Validate ingredient information on normal and order-reversed snapshot caches.

Supervision trains disposable probes only. Probe weights never enter the agent.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from .train import seed_everything


class Probe(nn.Module):
    def __init__(self, inputs: int, width: int = 64) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(inputs, width), nn.GELU(), nn.Linear(width, 2))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


def _load(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["snapshots"], payload["auxiliary_labels"], payload["metadata"]


def _fit(train_x, train_y, *, steps: int, batch_size: int, seed: int,
         device: torch.device):
    seed_everything(seed)
    model = Probe(train_x.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        indices = torch.randint(train_y.numel(), (batch_size,), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(
            model(train_x[indices].to(device)), train_y[indices].to(device))
        loss.backward()
        optimizer.step()
    return model.eval()


@torch.no_grad()
def _predict(model, x, batch_size, device):
    return torch.cat([
        model(x[offset:offset + batch_size].to(device)).argmax(-1).cpu()
        for offset in range(0, x.shape[0], batch_size)
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--test-cache", type=Path, required=True)
    parser.add_argument("--reversed-test-cache", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    train_x, train_y, train_metadata = _load(args.train_cache)
    test_x, test_y, test_metadata = _load(args.test_cache)
    reversed_x, reversed_y, reversed_metadata = _load(args.reversed_test_cache)
    if reversed_metadata.get("reverse_events") is not True:
        raise ValueError("reversed cache is not marked reverse_events=true")
    if test_x.shape != reversed_x.shape:
        raise ValueError("normal and reversed test caches must be aligned")

    results = {}
    predictions = {}
    for target, event_index, label_index, seed_offset in (
            ("first_identity", 0, 0, 0),
            ("rewarded_identity", 2, 1, 1)):
        source = train_x[:, event_index]
        mean = source.mean(0, keepdim=True)
        scale = source.std(0, keepdim=True).clamp_min(1e-5)
        train_features = (source - mean) / scale
        test_features = (test_x[:, event_index] - mean) / scale
        reversed_features = (reversed_x[:, event_index] - mean) / scale
        probe = _fit(
            train_features, train_y[:, label_index], steps=args.steps,
            batch_size=args.batch_size, seed=args.seed + seed_offset, device=device)
        normal_predictions = _predict(probe, test_features, args.batch_size, device)
        reversed_predictions = _predict(
            probe, reversed_features, args.batch_size, device)
        predictions[target] = (normal_predictions, reversed_predictions)
        results[target] = {
            "normal_accuracy": float(
                (normal_predictions == test_y[:, label_index]).float().mean()),
            "reversed_accuracy": float(
                (reversed_predictions == reversed_y[:, label_index]).float().mean()),
        }

    results["counterfactual_consistency"] = {
        "first_identity_prediction_flip_rate": float(
            (predictions["first_identity"][0]
             != predictions["first_identity"][1]).float().mean()),
        "rewarded_identity_prediction_same_rate": float(
            (predictions["rewarded_identity"][0]
             == predictions["rewarded_identity"][1]).float().mean()),
        "true_first_identity_flip_rate": float(
            (test_y[:, 0] != reversed_y[:, 0]).float().mean()),
        "true_rewarded_identity_same_rate": float(
            (test_y[:, 1] == reversed_y[:, 1]).float().mean()),
    }
    report = {
        "schema": "temporal-counterfactual-ingredient-probe-v1",
        "controller_frozen": True,
        "disposable_supervised_probe": True,
        "train_metadata": train_metadata,
        "test_metadata": test_metadata,
        "reversed_test_metadata": reversed_metadata,
        "results": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

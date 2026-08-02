"""Episodic nearest-neighbor probe over opaque agent trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .probe_nback_agent_context import RINGS, _features


def _knn(features: torch.Tensor, labels: torch.Tensor, *, k: int,
         shuffled: bool = False) -> float:
    split = int(features.shape[0] * 0.8)
    train_x, test_x = features[:split], features[split:]
    train_y, test_y = labels[:split], labels[split:]
    if shuffled:
        train_y = train_y[torch.randperm(train_y.shape[0], device=train_y.device)]
    mean = train_x.mean(0, keepdim=True)
    scale = train_x.std(0, keepdim=True).clamp_min(1e-4)
    train_x = (train_x - mean) / scale
    test_x = (test_x - mean) / scale
    predictions = []
    for row in test_x.split(128):
        distances = torch.cdist(row, train_x)
        nearest = distances.topk(k, largest=False).indices
        votes = train_y[nearest]
        predictions.append(torch.tensor([
            torch.bincount(sample.cpu(), minlength=len(RINGS)).argmax()
            for sample in votes
        ], device=features.device))
    prediction = torch.cat(predictions)
    return float((prediction == test_y).float().mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--count-per-ring", type=int, default=256)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=48800)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    device = torch.device(args.device)
    _, actions, rewards, labels = _features(
        args.checkpoint, count_per_ring=args.count_per_ring,
        trials=args.trials, seed=args.seed, device=device)
    records = torch.cat((
        actions.reshape(labels.shape[0], args.trials, -1),
        rewards.unsqueeze(-1)), dim=-1).flatten(1)
    permutation = torch.randperm(
        labels.shape[0], generator=torch.Generator(device=device).manual_seed(
            args.seed), device=device)
    records, labels = records[permutation], labels[permutation]
    results = {}
    for k in (1, 3, 5, 9):
        results[f"knn_{k}"] = _knn(records, labels, k=k)
    results["shuffled_knn_5"] = _knn(records, labels, k=5, shuffled=True)
    report = {
        "format": "nback_episode_retrieval_probe.v1", "rings": list(RINGS),
        "checkpoint": str(args.checkpoint), "count_per_ring": args.count_per_ring,
        "trials": args.trials, "seed": args.seed, "device": str(device),
        "results": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()

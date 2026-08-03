"""Diagnostic ceiling for an order-aware relational context encoder.

Ring labels are used only to train this disposable supervised probe. The
encoder itself receives opaque sensory/action/outcome records and explicitly
represents ordered event pairs; it is a capacity test before reward training,
not a capability claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from .probe_nback_agent_context import RINGS, _features, _fit


def _records(sensory: torch.Tensor, actions: torch.Tensor,
             rewards: torch.Tensor, trials: int) -> torch.Tensor:
    sensory = sensory.reshape(sensory.shape[0], trials, -1)
    actions = actions.reshape(actions.shape[0], trials, -1)
    return torch.cat((sensory, actions, rewards.unsqueeze(-1)), dim=-1)


def _relational_features(records: torch.Tensor) -> torch.Tensor:
    _, trials, width = records.shape
    pairs = []
    for first in range(trials):
        for second in range(first + 1, trials):
            left, right = records[:, first], records[:, second]
            distance = records.new_full((records.shape[0], 1),
                                        (second - first) / trials)
            pairs.append(torch.cat((left, right, left * right, distance), -1))
    return torch.stack(pairs, dim=1)


def _probe(records: torch.Tensor, labels: torch.Tensor, *, updates: int,
           seed: int) -> float:
    torch.manual_seed(seed)
    split = int(records.shape[0] * 0.8)
    train_records, test_records = records[:split], records[split:]
    train_labels, test_labels = labels[:split], labels[split:]
    pair_features = _relational_features(records)
    train_pair, test_pair = pair_features[:split], pair_features[split:]
    hidden = 96
    pair_encoder = nn.Sequential(
        nn.Linear(train_pair.shape[-1], 128), nn.GELU(),
        nn.Linear(128, hidden), nn.GELU())
    classifier = nn.Linear(hidden * 2, len(RINGS))
    model = nn.ModuleList((pair_encoder, classifier)).to(records.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3,
                                  weight_decay=1e-4)
    for _ in range(updates):
        encoded = pair_encoder(train_pair)
        pooled = torch.cat((encoded.mean(1), encoded.amax(1)), dim=-1)
        loss = nn.functional.cross_entropy(classifier(pooled), train_labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        encoded = pair_encoder(test_pair)
        pooled = torch.cat((encoded.mean(1), encoded.amax(1)), dim=-1)
        return float((classifier(pooled).argmax(-1) == test_labels).float().mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--count-per-ring", type=int, default=256)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--updates", type=int, default=256)
    parser.add_argument("--seed", type=int, default=49500)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    device = torch.device(args.device)
    sensory, actions, rewards, labels = _features(
        args.checkpoint, count_per_ring=args.count_per_ring,
        trials=args.trials, seed=args.seed, device=device)
    records = _records(sensory, actions, rewards, args.trials)
    permutation = torch.randperm(
        labels.shape[0], generator=torch.Generator(device=device).manual_seed(
            args.seed), device=device)
    records, actions, rewards, labels = (
        value[permutation]
        for value in (records, actions, rewards, labels))
    results = {
        "raw_action_reward": _fit(
            torch.cat((actions, rewards), dim=-1), labels,
            updates=args.updates, seed=args.seed + 1),
        "ordered_pair_encoder": _probe(
            records, labels, updates=args.updates, seed=args.seed + 2),
    }
    report = {
        "format": "nback_relational_context_ceiling.v1",
        "rings": list(RINGS), "checkpoint": str(args.checkpoint),
        "count_per_ring": args.count_per_ring, "trials": args.trials,
        "updates": args.updates, "seed": args.seed, "device": str(device),
        "results": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()

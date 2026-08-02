"""Self-supervised context representation probe.

The encoder sees only the frozen agent's opaque action/reward trajectory.  It is
trained to predict future scalar returns, then its latent is tested with a
throwaway task-context classifier.  The n-back label is used only for this
post-hoc diagnostic, never for the representation loss.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from .probe_nback_agent_context import RINGS, _features, _fit


def _self_supervised(records: torch.Tensor, labels: torch.Tensor, *, depth: int,
                     updates: int, seed: int,
                     shuffle_targets: bool = False) -> dict[str, float]:
    torch.manual_seed(seed)
    split = int(records.shape[0] * 0.8)
    train_records, test_records = records[:split], records[split:]
    train_labels, test_labels = labels[:split], labels[split:]
    input_width = records.shape[2] * depth
    latent = nn.Sequential(
        nn.Linear(input_width, 96), nn.GELU(), nn.Linear(96, 64), nn.GELU()
    ).to(records.device)
    future_width = (records.shape[1] - depth) * records.shape[2]
    predictor = nn.Linear(64, future_width).to(records.device)
    optimizer = torch.optim.AdamW(
        list(latent.parameters()) + list(predictor.parameters()),
        lr=3e-3, weight_decay=1e-4)
    train_x = train_records[:, :depth].flatten(1)
    train_y = train_records[:, depth:].reshape(train_records.shape[0], -1)
    if shuffle_targets:
        train_y = train_y[torch.randperm(
            train_y.shape[0], device=train_y.device)]
    for _ in range(updates):
        prediction = predictor(latent(train_x))
        # Predict future opaque actions and scalar rewards; the target head is
        # discarded after this representation-only training loop.
        loss = nn.functional.mse_loss(prediction.contiguous(), train_y.contiguous())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_latent = latent(train_x)
        test_latent = latent(test_records[:, :depth].flatten(1))
    probe = nn.Linear(train_latent.shape[1], len(RINGS)).to(records.device)
    probe_optimizer = torch.optim.AdamW(probe.parameters(), lr=3e-3)
    for _ in range(updates):
        loss = nn.functional.cross_entropy(probe(train_latent), train_labels)
        probe_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        probe_optimizer.step()
    with torch.no_grad():
        train_accuracy = (
            probe(train_latent).argmax(-1) == train_labels).float().mean()
        held_out_accuracy = (
            probe(test_latent).argmax(-1) == test_labels).float().mean()
    return {"train_accuracy": float(train_accuracy),
            "held_out_accuracy": float(held_out_accuracy)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--count-per-ring", type=int, default=256)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--depth", type=int, default=8, choices=(2, 4, 6, 8))
    parser.add_argument("--updates", type=int, default=256)
    parser.add_argument("--seed", type=int, default=48700)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    device = torch.device(args.device)
    sensory, actions, rewards, labels = _features(
        args.checkpoint, count_per_ring=args.count_per_ring,
        trials=args.trials, seed=args.seed, device=device)
    permutation = torch.randperm(
        labels.shape[0], generator=torch.Generator(device=device).manual_seed(
            args.seed), device=device)
    sensory, actions, rewards, labels = (
        value[permutation] for value in (sensory, actions, rewards, labels))
    action_sequence = actions.reshape(labels.shape[0], args.trials, -1)
    records = torch.cat((action_sequence, rewards.unsqueeze(-1)), dim=-1)
    results = {
        "raw_action_reward": _fit(
            records.flatten(1), labels, updates=args.updates,
            seed=args.seed + 1),
        "self_supervised_future_return": _self_supervised(
            records, labels, depth=args.depth, updates=args.updates,
            seed=args.seed + 2),
        "shuffled_return_control": _self_supervised(
            records, labels, depth=args.depth, updates=args.updates,
            seed=args.seed + 3, shuffle_targets=True),
    }
    report = {
        "format": "nback_self_supervised_context_probe.v1",
        "rings": list(RINGS), "checkpoint": str(args.checkpoint),
        "count_per_ring": args.count_per_ring, "trials": args.trials,
        "depth": args.depth, "updates": args.updates, "seed": args.seed,
        "device": str(device), "results": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()

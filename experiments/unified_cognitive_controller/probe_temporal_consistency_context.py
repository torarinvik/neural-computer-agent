"""Self-supervised temporal-consistency probe for task-context memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from .probe_nback_agent_context import RINGS, _features, _fit


def _records(sensory: torch.Tensor, actions: torch.Tensor,
             rewards: torch.Tensor, trials: int) -> torch.Tensor:
    sensory = sensory.reshape(sensory.shape[0], trials, -1)
    actions = actions.reshape(actions.shape[0], trials, -1)
    return torch.cat((sensory, actions, rewards.unsqueeze(-1)), dim=-1)


def _probe(records: torch.Tensor, labels: torch.Tensor, *, updates: int,
           seed: int, shuffled_labels: bool = False) -> float:
    torch.manual_seed(seed)
    split = int(records.shape[0] * 0.8)
    train, test = records[:split], records[split:]
    train_labels, test_labels = labels[:split], labels[split:]
    hidden = 64
    encoder = nn.GRU(records.shape[-1], hidden, batch_first=True).to(
        records.device)
    consistency_head = nn.Linear(hidden, 1).to(records.device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(consistency_head.parameters()),
        lr=3e-3, weight_decay=1e-4)
    generator = torch.Generator(device=records.device).manual_seed(seed + 17)
    permutations = torch.stack([
        torch.randperm(records.shape[1], generator=generator,
                       device=records.device)
        for _ in range(train.shape[0])])
    shuffled = torch.stack([
        train[index, permutations[index]] for index in range(train.shape[0])])
    train_views = torch.cat((train, shuffled), dim=0)
    consistency_targets = torch.cat((
        torch.ones(train.shape[0], device=records.device),
        torch.zeros(train.shape[0], device=records.device)))
    if shuffled_labels:
        consistency_targets = consistency_targets[torch.randperm(
            consistency_targets.shape[0], device=records.device)]
    for _ in range(updates):
        latent = encoder(train_views)[0][:, -1]
        loss = F.binary_cross_entropy_with_logits(
            consistency_head(latent).squeeze(-1), consistency_targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_latent = encoder(train)[0][:, -1]
        test_latent = encoder(test)[0][:, -1]
    probe = nn.Linear(hidden, len(RINGS)).to(records.device)
    probe_optimizer = torch.optim.AdamW(probe.parameters(), lr=3e-3)
    for _ in range(updates):
        loss = F.cross_entropy(probe(train_latent), train_labels)
        probe_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        probe_optimizer.step()
    with torch.no_grad():
        return float((probe(test_latent).argmax(-1) == test_labels).float().mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--count-per-ring", type=int, default=256)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--updates", type=int, default=256)
    parser.add_argument("--seed", type=int, default=49300)
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
    raw = _fit(torch.cat((actions, rewards), dim=-1), labels,
               updates=args.updates, seed=args.seed + 1)
    results = {
        "raw_action_reward": raw,
        "temporal_consistency": _probe(
            records, labels, updates=args.updates, seed=args.seed + 2),
        "shuffled_consistency_control": _probe(
            records, labels, updates=args.updates, seed=args.seed + 3,
            shuffled_labels=True),
    }
    report = {
        "format": "nback_temporal_consistency_context_probe.v1",
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

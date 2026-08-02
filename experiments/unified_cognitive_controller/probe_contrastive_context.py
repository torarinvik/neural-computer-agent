"""Contrastive-predictive probe for zero-label trajectory context."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from .probe_nback_agent_context import RINGS, _features, _fit


def _probe(records: torch.Tensor, labels: torch.Tensor, *, prefix_depth: int,
           updates: int, seed: int, shuffled_suffix: bool = False) -> float:
    torch.manual_seed(seed)
    split = int(records.shape[0] * 0.8)
    train_records, test_records = records[:split], records[split:]
    train_labels, test_labels = labels[:split], labels[split:]
    feature_width = records.shape[-1]
    suffix = train_records[:, prefix_depth:]
    if shuffled_suffix:
        suffix = suffix[torch.randperm(suffix.shape[0], device=suffix.device)]
    encoder = nn.Sequential(
        nn.Linear(prefix_depth * feature_width, 128), nn.GELU(),
        nn.Linear(128, 64)).to(records.device)
    target_encoder = nn.Sequential(
        nn.Linear(suffix.shape[1] * feature_width, 128), nn.GELU(),
        nn.Linear(128, 64)).to(records.device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(target_encoder.parameters()),
        lr=2e-3, weight_decay=1e-4)
    prefix = train_records[:, :prefix_depth].reshape(train_records.shape[0], -1)
    suffix_flat = suffix.reshape(suffix.shape[0], -1)
    positives = torch.arange(prefix.shape[0], device=records.device)
    for _ in range(updates):
        query = F.normalize(encoder(prefix), dim=-1)
        key = F.normalize(target_encoder(suffix_flat), dim=-1)
        logits = query @ key.transpose(0, 1) / 0.1
        loss = F.cross_entropy(logits, positives)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_latent = encoder(prefix)
        test_latent = encoder(
            test_records[:, :prefix_depth].reshape(test_records.shape[0], -1))
    probe = nn.Linear(train_latent.shape[-1], len(RINGS)).to(records.device)
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
    parser.add_argument("--prefix-depth", type=int, default=5,
                        choices=(2, 4, 5, 6))
    parser.add_argument("--updates", type=int, default=256)
    parser.add_argument("--seed", type=int, default=48900)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    device = torch.device(args.device)
    _, actions, rewards, labels = _features(
        args.checkpoint, count_per_ring=args.count_per_ring,
        trials=args.trials, seed=args.seed, device=device)
    records = torch.cat((
        actions.reshape(labels.shape[0], args.trials, -1),
        rewards.unsqueeze(-1)), dim=-1)
    permutation = torch.randperm(
        labels.shape[0], generator=torch.Generator(device=device).manual_seed(
            args.seed), device=device)
    records, labels = records[permutation], labels[permutation]
    results = {
        "raw_action_reward": _fit(
            records.flatten(1), labels, updates=args.updates,
            seed=args.seed + 1),
        "contrastive_predictive": _probe(
            records, labels, prefix_depth=args.prefix_depth,
            updates=args.updates, seed=args.seed + 2),
        "shuffled_suffix_control": _probe(
            records, labels, prefix_depth=args.prefix_depth,
            updates=args.updates, seed=args.seed + 3, shuffled_suffix=True),
    }
    report = {
        "format": "nback_contrastive_context_probe.v1", "rings": list(RINGS),
        "checkpoint": str(args.checkpoint), "count_per_ring": args.count_per_ring,
        "trials": args.trials, "prefix_depth": args.prefix_depth,
        "updates": args.updates, "seed": args.seed, "device": str(device),
        "results": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()

"""Zero-label view-invariant trajectory representation probe.

Two independently masked views of one opaque trajectory are treated as a
positive pair.  No n-back/ring label enters the representation objective; the
label is used only by a disposable held-out probe after training.
"""

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


def _view(record: torch.Tensor, *, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=record.device).manual_seed(seed)
    mask = torch.rand(record.shape, generator=generator, device=record.device)
    # Keep the outcome channel intact; it is the only direct success signal.
    mask[..., :-1] = (mask[..., :-1] > 0.15).float()
    view = record * mask
    # Small noise prevents exact sequence memorization while preserving the
    # discrete/opaque nature of the observed stream.
    noise = torch.randn(record.shape[:-1] + (record.shape[-1] - 1,),
                        generator=generator, device=record.device) * 0.01
    view[..., :-1] = view[..., :-1] + noise
    return view


def _probe(records: torch.Tensor, labels: torch.Tensor, *, updates: int,
           seed: int, shuffled_pairs: bool = False) -> float:
    torch.manual_seed(seed)
    split = int(records.shape[0] * 0.8)
    train, test = records[:split], records[split:]
    train_labels, test_labels = labels[:split], labels[split:]
    hidden = 64
    encoder = nn.GRU(records.shape[-1], hidden, batch_first=True).to(
        records.device)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=3e-3,
                                  weight_decay=1e-4)
    view_a = _view(train, seed=seed + 11)
    view_b = _view(train, seed=seed + 12)
    if shuffled_pairs:
        view_b = view_b[torch.randperm(view_b.shape[0], device=records.device)]
    positives = torch.arange(train.shape[0], device=records.device)
    for _ in range(updates):
        query = F.normalize(encoder(view_a)[0][:, -1], dim=-1)
        key = F.normalize(encoder(view_b)[0][:, -1], dim=-1)
        loss = F.cross_entropy(query @ key.transpose(0, 1) / 0.1, positives)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_latent = encoder(view_a)[0][:, -1]
        test_latent = encoder(_view(test, seed=seed + 13))[0][:, -1]
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
    parser.add_argument("--seed", type=int, default=49100)
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
    records, labels = records[permutation], labels[permutation]
    results = {
        "raw_action_reward": _fit(
            records.flatten(1), labels, updates=args.updates,
            seed=args.seed + 1),
        "view_invariant": _probe(records, labels, updates=args.updates,
                                  seed=args.seed + 2),
        "shuffled_pair_control": _probe(
            records, labels, updates=args.updates, seed=args.seed + 3,
            shuffled_pairs=True),
    }
    report = {
        "format": "nback_view_invariant_context_probe.v1",
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

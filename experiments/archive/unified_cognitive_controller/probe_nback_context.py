"""Diagnostic: is the current n-back task identifiable from experience?

This is a disposable verifier-side probe, not an agent component.  It tests a
necessary condition for a continual one-controller claim: if the sensory
sequence and/or the observed feedback do not identify which relation is being
trained, a shared answer path cannot reliably select among n-back skills.
Labels and targets are used only for this localization measurement.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn

from .brainworkshop_gym import BrainWorkshopConfig, generate_brainworkshop_episode


RINGS = (1, 5, 6, 7, 8)


def _examples(count_per_ring: int, *, seed: int, trials: int,
              device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    sensory = []
    feedback = []
    labels = []
    for ring_index, n_back in enumerate(RINGS):
        for index in range(count_per_ring):
            episode = generate_brainworkshop_episode(
                BrainWorkshopConfig(
                    n_back=n_back, trials=trials, balanced_matches=False,
                    modalities=("text",), text_vocab=8),
                seed=seed + ring_index * 100_000 + index,
                device=device)
            tokens = [stimulus.text for stimulus in episode.stimuli]
            targets = [int(target & 4 != 0)
                       for target in episode.verifier_targets()]
            sensory.append(torch.nn.functional.one_hot(
                torch.tensor(tokens, device=device), num_classes=8).flatten()
                .float())
            feedback.append(torch.tensor(targets, device=device).float())
            labels.append(ring_index)
    return (torch.stack(sensory), torch.stack(feedback),
            torch.tensor(labels, device=device))


def _fit(features: torch.Tensor, labels: torch.Tensor, *, updates: int,
         seed: int) -> dict[str, float]:
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(features.shape[1], 96), nn.GELU(),
        nn.Linear(96, 64), nn.GELU(), nn.Linear(64, len(RINGS)),
    ).to(features.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    split = int(features.shape[0] * 0.8)
    train_x, test_x = features[:split], features[split:]
    train_y, test_y = labels[:split], labels[split:]
    for _ in range(updates):
        logits = model(train_x)
        loss = nn.functional.cross_entropy(logits, train_y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_accuracy = (model(train_x).argmax(-1) == train_y).float().mean()
        test_accuracy = (model(test_x).argmax(-1) == test_y).float().mean()
    return {"train_accuracy": float(train_accuracy),
            "held_out_accuracy": float(test_accuracy)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--count-per-ring", type=int, default=1024)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--updates", type=int, default=256)
    parser.add_argument("--seed", type=int, default=48400)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    sensory, feedback, labels = _examples(
        args.count_per_ring, seed=args.seed, trials=args.trials, device=device)
    permutation = torch.randperm(labels.shape[0], generator=torch.Generator(
        device=device).manual_seed(args.seed), device=device)
    sensory, feedback, labels = sensory[permutation], feedback[permutation], labels[permutation]
    shuffled = labels[torch.randperm(
        labels.shape[0], generator=torch.Generator(device=device).manual_seed(
            args.seed + 1), device=device)]
    results = {
        "sensory_only": _fit(sensory, labels, updates=args.updates,
                              seed=args.seed + 2),
        "sensory_plus_feedback": _fit(
            torch.cat((sensory, feedback), dim=-1), labels,
            updates=args.updates, seed=args.seed + 3),
        "shuffled_label_control": _fit(
            sensory, shuffled, updates=args.updates, seed=args.seed + 4),
    }
    report = {
        "format": "nback_context_probe.v1", "rings": list(RINGS),
        "count_per_ring": args.count_per_ring, "trials": args.trials,
        "updates": args.updates, "seed": args.seed, "device": str(device),
        "results": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()

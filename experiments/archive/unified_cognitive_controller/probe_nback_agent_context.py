"""Probe task-context information in the frozen agent's actual experience.

The classifier is disposable and verifier-side.  It receives only rendered
text features plus the frozen policy's opaque actions and scalar rewards; it
never receives the n-back label or verifier target.  A held-out result above
chance is evidence that a trajectory-context memory could select skills.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from .audit_nback_checkpoint_retention import _load_policy
from .brainworkshop_gym import BrainWorkshopConfig, generate_brainworkshop_episode
from .train_brainworkshop_policy import _rollout


RINGS = (1, 5, 6, 7, 8)


def _features(checkpoint: Path, *, count_per_ring: int, trials: int,
              seed: int, device: torch.device,
              ring_values: tuple[int, ...] = RINGS):
    policy = _load_policy(checkpoint, device)
    sensory, actions, rewards, labels = [], [], [], []
    for ring_index, n_back in enumerate(ring_values):
        config = BrainWorkshopConfig(
            n_back=n_back, trials=trials, balanced_matches=False,
            modalities=("vision", "audio", "text"), text_vocab=8,
            trial_ms=1000)
        batch_seed = seed + ring_index * 100_000
        rollout = _rollout(
            policy, config, batch_size=count_per_ring, seed=batch_seed,
            device=device, sample=False, external_history=True,
            per_stream_external_history=True, external_history_depth=8)
        tokens = []
        for index in range(count_per_ring):
            episode = generate_brainworkshop_episode(
                config, seed=batch_seed + index, device=device)
            tokens.append([stimulus.text for stimulus in episode.stimuli])
        sensory.append(torch.nn.functional.one_hot(
            torch.tensor(tokens, device=device), num_classes=8).flatten(1)
            .float())
        action_count = 1 << len(policy.action_bits)
        actions.append(torch.nn.functional.one_hot(
            rollout.actions.transpose(0, 1).clamp_max(action_count - 1),
            num_classes=action_count).flatten(1).float())
        rewards.append(rollout.rewards.transpose(0, 1).float())
        labels.append(torch.full((count_per_ring,), ring_index,
                                 dtype=torch.long, device=device))
    return (torch.cat(sensory), torch.cat(actions), torch.cat(rewards),
            torch.cat(labels))


def _fit(features: torch.Tensor, labels: torch.Tensor, *, updates: int,
         seed: int, num_classes: int = len(RINGS)) -> dict[str, float]:
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(features.shape[1], 128), nn.GELU(),
        nn.Linear(128, 64), nn.GELU(), nn.Linear(64, num_classes),
    ).to(features.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3,
                                  weight_decay=1e-4)
    split = int(features.shape[0] * 0.8)
    train_x, test_x = features[:split], features[split:]
    train_y, test_y = labels[:split], labels[split:]
    for _ in range(updates):
        loss = nn.functional.cross_entropy(model(train_x), train_y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_accuracy = (model(train_x).argmax(-1) == train_y).float().mean()
        held_out_accuracy = (model(test_x).argmax(-1) == test_y).float().mean()
    return {"train_accuracy": float(train_accuracy),
            "held_out_accuracy": float(held_out_accuracy)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--count-per-ring", type=int, default=256)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--updates", type=int, default=256)
    parser.add_argument("--seed", type=int, default=48500)
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
    results = {
        "sensory_only": _fit(sensory, labels, updates=args.updates,
                              seed=args.seed + 1),
        "action_reward_only": _fit(
            torch.cat((actions, rewards), dim=-1), labels,
            updates=args.updates, seed=args.seed + 2),
        "sensory_action_reward": _fit(
            torch.cat((sensory, actions, rewards), dim=-1), labels,
            updates=args.updates, seed=args.seed + 3),
    }
    report = {
        "format": "nback_agent_context_probe.v1", "rings": list(RINGS),
        "checkpoint": str(args.checkpoint),
        "count_per_ring": args.count_per_ring, "trials": args.trials,
        "updates": args.updates, "seed": args.seed, "device": str(device),
        "results": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()

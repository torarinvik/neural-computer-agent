"""Self-supervised masked-event probe for task-context representations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from .probe_nback_agent_context import RINGS, _features, _fit


def _inputs(sensory: torch.Tensor, actions: torch.Tensor,
            rewards: torch.Tensor, trials: int):
    sensory = sensory.reshape(sensory.shape[0], trials, -1)
    actions = actions.reshape(actions.shape[0], trials, -1)
    previous_actions = torch.cat(
        (torch.zeros_like(actions[:, :1]), actions[:, :-1]), dim=1)
    previous_rewards = torch.cat(
        (torch.zeros_like(rewards[:, :1]), rewards[:, :-1]), dim=1)
    inputs = torch.cat((sensory, previous_actions,
                        previous_rewards.unsqueeze(-1)), dim=-1)
    action_targets = actions.argmax(-1)
    reward_targets = (rewards > 0).float()
    return inputs, action_targets, reward_targets


def _probe(inputs: torch.Tensor, action_targets: torch.Tensor,
           reward_targets: torch.Tensor, labels: torch.Tensor, *, updates: int,
           seed: int, shuffled_targets: bool = False) -> float:
    torch.manual_seed(seed)
    split = int(inputs.shape[0] * 0.8)
    train_x, test_x = inputs[:split], inputs[split:]
    train_a, test_a = action_targets[:split], action_targets[split:]
    train_r, test_r = reward_targets[:split], reward_targets[split:]
    train_y, test_y = labels[:split], labels[split:]
    hidden = 64
    action_count = int(action_targets.max().item()) + 1
    encoder = nn.GRU(inputs.shape[-1], hidden, batch_first=True).to(
        inputs.device)
    action_head = nn.Linear(hidden, action_count).to(inputs.device)
    reward_head = nn.Linear(hidden, 1).to(inputs.device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(action_head.parameters())
        + list(reward_head.parameters()), lr=3e-3, weight_decay=1e-4)
    if shuffled_targets:
        permutation = torch.randperm(train_x.shape[0], device=inputs.device)
        train_a, train_r = train_a[permutation], train_r[permutation]
    for _ in range(updates):
        states, _ = encoder(train_x)
        action_logits = action_head(states)
        reward_logits = reward_head(states).squeeze(-1)
        loss = F.cross_entropy(action_logits.flatten(0, 1), train_a.flatten())
        loss = loss + F.binary_cross_entropy_with_logits(
            reward_logits[:, 1:], train_r[:, 1:])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_latent = encoder(train_x)[0][:, -1]
        test_latent = encoder(test_x)[0][:, -1]
    probe = nn.Linear(hidden, len(RINGS)).to(inputs.device)
    probe_optimizer = torch.optim.AdamW(probe.parameters(), lr=3e-3)
    for _ in range(updates):
        loss = F.cross_entropy(probe(train_latent), train_y)
        probe_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        probe_optimizer.step()
    with torch.no_grad():
        return float((probe(test_latent).argmax(-1) == test_y).float().mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--count-per-ring", type=int, default=256)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--updates", type=int, default=256)
    parser.add_argument("--seed", type=int, default=49200)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    device = torch.device(args.device)
    sensory, actions, rewards, labels = _features(
        args.checkpoint, count_per_ring=args.count_per_ring,
        trials=args.trials, seed=args.seed, device=device)
    inputs, action_targets, reward_targets = _inputs(
        sensory, actions, rewards, args.trials)
    permutation = torch.randperm(
        labels.shape[0], generator=torch.Generator(device=device).manual_seed(
            args.seed), device=device)
    inputs, actions, rewards, action_targets, reward_targets, labels = (
        value[permutation]
        for value in (inputs, actions, rewards, action_targets,
                      reward_targets, labels))
    records = torch.cat((actions, rewards), dim=-1)
    results = {
        "raw_action_reward": _fit(
            records, labels, updates=args.updates, seed=args.seed + 1),
        "masked_event_predictive": _probe(
            inputs, action_targets, reward_targets, labels,
            updates=args.updates, seed=args.seed + 2),
        "shuffled_event_control": _probe(
            inputs, action_targets, reward_targets, labels,
            updates=args.updates, seed=args.seed + 3, shuffled_targets=True),
    }
    report = {
        "format": "nback_masked_event_context_probe.v1",
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

"""Zero-label predictive-state probe using the next observed outcome.

The probe trains only on the stream the agent could observe: sensory token,
opaque action, and the previous scalar outcome.  It predicts the next scalar
outcome, then discards that head and asks whether the final recurrent state
retains task identity.  Ring labels are used only by a disposable evaluation
probe; they never enter the predictive objective.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from .probe_nback_agent_context import RINGS, _features, _fit


def _build_sequences(sensory: torch.Tensor, actions: torch.Tensor,
                     rewards: torch.Tensor, trials: int) -> tuple[torch.Tensor,
                                                                  torch.Tensor]:
    sensory = sensory.reshape(sensory.shape[0], trials, -1)
    actions = actions.reshape(actions.shape[0], trials, -1)
    # At decision time the current action and outcome are not yet known.
    # The predictor therefore receives current sensory input plus the previous
    # opaque action/outcome, and predicts the current observed outcome.
    previous_actions = torch.cat(
        (torch.zeros_like(actions[:, :1]), actions[:, :-1]), dim=1)
    previous_rewards = torch.cat(
        (torch.zeros_like(rewards[:, :1]), rewards[:, :-1]), dim=1)
    inputs = torch.cat((sensory, previous_actions,
                        previous_rewards.unsqueeze(-1)), dim=-1)
    return inputs, rewards.unsqueeze(-1)


def _predictive_probe(sequences: torch.Tensor, outcomes: torch.Tensor,
                     labels: torch.Tensor, *,
                     updates: int, seed: int,
                     shuffled_targets: bool = False) -> float:
    torch.manual_seed(seed)
    split = int(sequences.shape[0] * 0.8)
    train_x, test_x = sequences[:split], sequences[split:]
    train_y, test_y = labels[:split], labels[split:]
    hidden = 64
    encoder = nn.GRU(sequences.shape[-1], hidden, batch_first=True).to(
        sequences.device)
    head = nn.Sequential(nn.Linear(hidden, 32), nn.GELU(), nn.Linear(32, 1)).to(
        sequences.device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(head.parameters()),
        lr=3e-3, weight_decay=1e-4)
    # The verifier's signed score is converted to an observed success bit;
    # this is still an unlabeled sensor stream from the agent's perspective.
    target = (outcomes[:split] > 0).float().detach()
    if shuffled_targets:
        target = target[torch.randperm(target.shape[0], device=target.device)]
    for _ in range(updates):
        states, _ = encoder(train_x)
        prediction = head(states[:, 1:])
        loss = F.binary_cross_entropy_with_logits(
            prediction, target[:, 1:])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_latent = encoder(train_x)[0][:, -1]
        test_latent = encoder(test_x)[0][:, -1]
    probe = nn.Linear(hidden, len(RINGS)).to(sequences.device)
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
    parser.add_argument("--seed", type=int, default=49000)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    device = torch.device(args.device)
    sensory, actions, rewards, labels = _features(
        args.checkpoint, count_per_ring=args.count_per_ring,
        trials=args.trials, seed=args.seed, device=device)
    sequences, outcomes = _build_sequences(
        sensory, actions, rewards, args.trials)
    permutation = torch.randperm(
        labels.shape[0], generator=torch.Generator(device=device).manual_seed(
            args.seed), device=device)
    sequences, outcomes, labels = (value[permutation]
                                   for value in (sequences, outcomes, labels))
    raw = _fit(sequences.flatten(1), labels, updates=args.updates,
               seed=args.seed + 1)
    results = {
        "raw_action_reward": raw,
        "next_outcome_predictive": _predictive_probe(
            sequences, outcomes, labels, updates=args.updates,
            seed=args.seed + 2),
        "shuffled_next_outcome_control": _predictive_probe(
            sequences, outcomes, labels, updates=args.updates,
            seed=args.seed + 3, shuffled_targets=True),
    }
    report = {
        "format": "nback_next_outcome_context_probe.v1",
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

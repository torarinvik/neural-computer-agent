"""Supervised ceiling probe for an ordered recurrent relational encoder."""

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


class RecurrentRelationalEncoder(nn.Module):
    """Causal event reader with a learned relation-weighted RAM read."""

    def __init__(self, input_width: int, hidden: int = 64,
                 max_history: int = 10) -> None:
        super().__init__()
        self.hidden = hidden
        self.input_projection = nn.Linear(input_width, hidden)
        self.position = nn.Parameter(torch.randn(max_history, hidden) * 0.02)
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.key = nn.Linear(hidden * 3, hidden, bias=False)
        self.value = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.relation = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.update = nn.GRUCell(hidden * 2, hidden)

    def forward(self, records: torch.Tensor) -> torch.Tensor:
        events = self.input_projection(records)
        state = events.new_zeros((events.shape[0], self.hidden))
        previous: list[torch.Tensor] = []
        for index in range(events.shape[1]):
            current = events[:, index]
            if previous:
                history = torch.stack(previous, dim=1)
                current_expanded = current.unsqueeze(1).expand_as(history)
                relation = torch.cat((current_expanded, history,
                                      current_expanded * history), dim=-1)
                relation = relation + torch.cat((
                    torch.zeros_like(relation[..., :-self.hidden]),
                    self.position[:index].unsqueeze(0).expand(
                        relation.shape[0], -1, -1)), dim=-1)
                relation = self.relation(relation)
                # The relation MLP contributes to both addressing and value;
                # the current event can only read prior events.
                address = self.key(torch.cat((
                    current_expanded, history, current_expanded * history), -1))
                query = self.query(current).unsqueeze(1)
                weights = F.softmax(
                    (query * address).sum(-1) / self.hidden**0.5, dim=-1)
                context = (weights.unsqueeze(-1) *
                           self.value(torch.cat((
                               history, relation, history * relation), -1))).sum(1)
            else:
                context = torch.zeros_like(current)
            state = self.update(torch.cat((current, context), -1), state)
            previous.append(current)
        return state


def _probe(records: torch.Tensor, labels: torch.Tensor, *, updates: int,
           seed: int, shuffled_labels: bool = False) -> float:
    torch.manual_seed(seed)
    split = int(records.shape[0] * 0.8)
    train, test = records[:split], records[split:]
    train_labels, test_labels = labels[:split], labels[split:]
    if shuffled_labels:
        train_labels = train_labels[torch.randperm(
            train_labels.shape[0], device=records.device)]
    encoder = RecurrentRelationalEncoder(records.shape[-1]).to(records.device)
    classifier = nn.Linear(encoder.hidden, len(RINGS)).to(records.device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=3e-3, weight_decay=1e-4)
    for _ in range(updates):
        latent = encoder(train)
        loss = F.cross_entropy(classifier(latent), train_labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return float((classifier(encoder(test)).argmax(-1) == test_labels)
                     .float().mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--count-per-ring", type=int, default=256)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--updates", type=int, default=256)
    parser.add_argument("--seed", type=int, default=49600)
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
        "recurrent_relational_encoder": _probe(
            records, labels, updates=args.updates, seed=args.seed + 2),
        "shuffled_label_control": _probe(
            records, labels, updates=args.updates, seed=args.seed + 3,
            shuffled_labels=True),
    }
    report = {
        "format": "nback_recurrent_relational_context_ceiling.v1",
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

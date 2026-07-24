"""Learn task-agnostic arbitration between legacy and event-indexed readers."""
from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import torch
from torch import nn

from .environment import (
    generate_attention_lifetime,
    generate_compositional_temporal_attention_lifetime,
    generate_shape_attention_lifetime)
from .probe_temporal_rule_memory import _load
from .train import _forward, seed_everything
from .train_consolidator import _initial_memory


GENERATORS = {
    "compositional_temporal": partial(
        generate_compositional_temporal_attention_lifetime,
        feedback_mode="color-button"),
    "spatial": generate_attention_lifetime,
    "shape": generate_shape_attention_lifetime,
}


@torch.no_grad()
def _collect(model, generator, *, start, lifetimes, batch_size, device):
    base_parts, reader_parts, target_parts = [], [], []
    density_parts = []
    for offset in range(0, lifetimes, batch_size):
        count = min(batch_size, lifetimes - offset)
        items = [
            generator(start + offset + index, heldout=True, query_count=2)
            for index in range(count)]
        memory = _initial_memory(model, items, device)
        for query_index in range(2):
            captured = []
            handle = model.event_indexed_memory_reader.register_forward_hook(
                lambda _module, inputs, output: captured.append((
                    tuple(value.detach() for value in inputs),
                    output.detach())))
            output, targets = _forward(
                model,
                [item.old_audit_queries[query_index] for item in items],
                memory, device)
            handle.remove()
            (rows, query), reader_logits = captured[-1]
            reader = model.event_indexed_memory_reader
            z_rows = (
                rows - reader.rows_mean) / reader.rows_scale.clamp_min(1e-5)
            z_query = (
                query - reader.query_mean) / reader.query_scale.clamp_min(1e-5)
            density_parts.append(torch.stack((
                z_rows.square().mean((1, 2)),
                z_query.square().mean(1)), dim=-1).cpu())
            base_parts.append(output.answer_logits[:, -1].cpu())
            reader_parts.append(reader_logits.cpu())
            target_parts.append(targets.cpu())
    return {
        "base": torch.cat(base_parts),
        "reader": torch.cat(reader_parts),
        "density": torch.cat(density_parts),
        "target": torch.cat(target_parts),
    }


def _features(data):
    base_prob = torch.softmax(data["base"], dim=-1)
    reader_prob = torch.softmax(data["reader"], dim=-1)
    base_entropy = -(
        base_prob * base_prob.clamp_min(1e-8).log()).sum(-1, keepdim=True)
    reader_entropy = -(
        reader_prob * reader_prob.clamp_min(1e-8).log()).sum(-1, keepdim=True)
    return torch.cat((
        data["base"], data["reader"], data["density"],
        base_entropy, reader_entropy), dim=-1)


class ReaderArbitrator(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(features), nn.Linear(features, 32), nn.GELU(),
            nn.Linear(32, 1))
        self.log_base_scale = nn.Parameter(torch.zeros(()))
        self.log_reader_scale = nn.Parameter(torch.zeros(()))

    def forward(self, features, base, reader):
        gate = torch.sigmoid(self.gate(features))
        logits = (
            (1 - gate) * self.log_base_scale.exp() * base +
            gate * self.log_reader_scale.exp() * reader)
        return logits, gate


def _merge(parts):
    return {
        key: torch.cat([part[key] for part in parts])
        for key in parts[0]}


def _fit(train_by_task, test_by_task, *, seed, device,
         shuffle_labels=False):
    seed_everything(seed)
    train = _merge(list(train_by_task.values()))
    model = ReaderArbitrator(_features(train).shape[-1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-3, weight_decay=1e-3)
    x = _features(train).to(device)
    base = train["base"].to(device)
    reader = train["reader"].to(device)
    target = train["target"].to(device)
    if shuffle_labels:
        generator = torch.Generator().manual_seed(seed + 991)
        target = target[torch.randperm(
            target.numel(), generator=generator).to(device)]
    for _ in range(400):
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x, base, reader)
        loss = nn.functional.cross_entropy(logits, target)
        loss.backward()
        optimizer.step()
    result = {}
    with torch.no_grad():
        for task, data in test_by_task.items():
            logits, gate = model(
                _features(data).to(device), data["base"].to(device),
                data["reader"].to(device))
            target = data["target"].to(device)
            result[task] = {
                "base_accuracy": float((
                    data["base"].argmax(-1) == data["target"]).float().mean()),
                "reader_accuracy": float((
                    data["reader"].argmax(-1) == data["target"]).float().mean()),
                "gated_accuracy": float((
                    logits.argmax(-1) == target).float().mean()),
                "mean_reader_gate": float(gate.mean()),
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-checkpoint", type=Path, required=True)
    parser.add_argument("--consolidator-checkpoint", type=Path, required=True)
    parser.add_argument("--pairwise-transfer-checkpoint", type=Path, required=True)
    parser.add_argument("--projection-transfer-checkpoint", type=Path, required=True)
    parser.add_argument("--transfer-strength", type=float, default=.01)
    parser.add_argument("--train-lifetimes", type=int, default=256)
    parser.add_argument("--test-lifetimes", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=673)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    transfers = (
        str(args.pairwise_transfer_checkpoint),
        str(args.projection_transfer_checkpoint))
    model, _ = _load(
        args.controller_checkpoint, args.consolidator_checkpoint, device,
        transfer_paths=transfers, transfer_strength=args.transfer_strength)
    train, test = {}, {}
    for index, (task, generator) in enumerate(GENERATORS.items()):
        train[task] = _collect(
            model, generator, start=121_000_000 + index * 1_000_000,
            lifetimes=args.train_lifetimes, batch_size=args.batch_size,
            device=device)
        test[task] = _collect(
            model, generator, start=125_000_000 + index * 1_000_000,
            lifetimes=args.test_lifetimes, batch_size=args.batch_size,
            device=device)
    result = {
        "schema": "event-reader-arbitration-probe-v1",
        "agent_and_readers_frozen": True,
        "sensory_only": True,
        "no_task_identity_input": True,
        "balanced_examples_per_task": True,
        "train_lifetimes_per_task": args.train_lifetimes,
        "test_lifetimes_per_task": args.test_lifetimes,
        "arbitrator": _fit(
            train, test, seed=args.seed, device=device),
        "shuffled_label_control": _fit(
            train, test, seed=args.seed, device=device,
            shuffle_labels=True),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

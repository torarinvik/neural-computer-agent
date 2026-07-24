"""Measure task-agnostic input density for the event-indexed reader."""
from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import torch

from .environment import (
    generate_attention_lifetime,
    generate_compositional_temporal_attention_lifetime,
    generate_shape_attention_lifetime,
    generate_temporal_attention_lifetime)
from .probe_temporal_rule_memory import _load
from .train import _forward, seed_everything
from .train_consolidator import _initial_memory


GENERATORS = {
    "temporal": partial(
        generate_temporal_attention_lifetime,
        feedback_mode="color-button"),
    "compositional_temporal": partial(
        generate_compositional_temporal_attention_lifetime,
        feedback_mode="color-button"),
    "spatial": generate_attention_lifetime,
    "shape": generate_shape_attention_lifetime,
}


@torch.no_grad()
def _collect(model, generator, *, start, lifetimes, batch_size, device):
    records = []
    for offset in range(0, lifetimes, batch_size):
        count = min(batch_size, lifetimes - offset)
        items = [
            generator(
                start + offset + index, heldout=True, query_count=1)
            for index in range(count)]
        memory = _initial_memory(model, items, device)
        captured = []
        handle = model.event_indexed_memory_reader.register_forward_pre_hook(
            lambda _module, inputs: captured.append(tuple(
                value.detach() for value in inputs)))
        _forward(
            model, [item.old_audit_queries[0] for item in items],
            memory, device)
        handle.remove()
        rows, query = captured[-1]
        reader = model.event_indexed_memory_reader
        rows = (
            rows - reader.rows_mean) / reader.rows_scale.clamp_min(1e-5)
        query = (
            query - reader.query_mean) / reader.query_scale.clamp_min(1e-5)
        row_distance = rows.square().mean((1, 2))
        query_distance = query.square().mean(1)
        records.append(torch.stack((
            row_distance, query_distance,
            torch.maximum(row_distance, query_distance)), dim=-1).cpu())
    return torch.cat(records)


def _summary(values):
    quantiles = torch.tensor([0, .1, .25, .5, .75, .9, 1.0])
    return {
        "columns": [
            "row_density_distance", "query_density_distance",
            "max_density_distance"],
        "mean": [float(value) for value in values.mean(0)],
        "quantiles": {
            str(float(q)): [
                float(value) for value in torch.quantile(values, q, dim=0)]
            for q in quantiles},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-checkpoint", type=Path, required=True)
    parser.add_argument("--consolidator-checkpoint", type=Path, required=True)
    parser.add_argument("--pairwise-transfer-checkpoint", type=Path, required=True)
    parser.add_argument("--projection-transfer-checkpoint", type=Path, required=True)
    parser.add_argument("--transfer-strength", type=float, default=.01)
    parser.add_argument("--lifetimes", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=667)
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
    result = {"schema": "event-indexed-reader-ood-probe-v1"}
    for index, (name, generator) in enumerate(GENERATORS.items()):
        result[name] = _summary(_collect(
            model, generator, start=117_000_000 + index * 1_000_000,
            lifetimes=args.lifetimes, batch_size=args.batch_size,
            device=device))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

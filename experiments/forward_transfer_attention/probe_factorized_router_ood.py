"""Measure task-agnostic density/confidence signals for router activation."""
from __future__ import annotations

import argparse
import json
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
    "temporal": generate_temporal_attention_lifetime,
    "compositional_temporal":
        generate_compositional_temporal_attention_lifetime,
    "spatial": generate_attention_lifetime,
    "shape": generate_shape_attention_lifetime,
}


def _collect(model, consolidator, generator, *, start, lifetimes,
             batch_size, device):
    records = []
    for offset in range(0, lifetimes, batch_size):
        count = min(batch_size, lifetimes - offset)
        kwargs = {"heldout": True, "query_count": 1}
        if generator in (
                generate_temporal_attention_lifetime,
                generate_compositional_temporal_attention_lifetime):
            kwargs["feedback_mode"] = "color-button"
        items = [
            generator(start + offset + index, **kwargs)
            for index in range(count)]
        memory = _initial_memory(model, items, device)
        first_raw = None
        for shot in range(2):
            output, _ = _forward(
                model, [item.supports[shot] for item in items], memory, device)
            if first_raw is None:
                first_raw = (
                    output.write_keys, output.write_values,
                    output.write_strengths)
            memory = memory.append(
                output.write_keys, output.write_values,
                output.write_strengths,
                torch.ones_like(output.write_strengths))
            memory = consolidator(memory)
            memory = memory.append(
                first_raw[0], first_raw[1], first_raw[2],
                torch.ones_like(first_raw[2]))
        captured = []
        handle = model.latest_row_factorized_router.register_forward_hook(
            lambda _module, inputs, output: captured.append((
                tuple(value.detach() for value in inputs),
                {key: value.detach() for key, value in output.items()})))
        with torch.no_grad():
            _forward(model, [item.future_queries[0] for item in items],
                     memory, device)
        handle.remove()
        inputs, output = captured[model.thought_steps - 1]
        support, first, second = inputs
        router = model.latest_row_factorized_router
        z_support = (
            support - router.support_mean) / router.support_scale.clamp_min(1e-5)
        z_first = (
            first - router.first_mean) / router.first_scale.clamp_min(1e-5)
        z_second = (
            second - router.second_mean) / router.second_scale.clamp_min(1e-5)
        density_distance = torch.cat(
            (z_support, z_first, z_second), dim=-1).square().mean(-1)
        route_confidence = output["route"].max(-1).values
        rule_confidence = torch.softmax(
            output["rule"], dim=-1).max(-1).values
        candidate_confidence = torch.stack((
            torch.softmax(output["first_action"], dim=-1).max(-1).values,
            torch.softmax(output["second_action"], dim=-1).max(-1).values),
            dim=-1).mean(-1)
        records.append(torch.stack((
            density_distance, route_confidence,
            rule_confidence, candidate_confidence), dim=-1).cpu())
    return torch.cat(records)


def _summary(values):
    quantiles = torch.tensor([0, .1, .25, .5, .75, .9, 1.0])
    return {
        "mean": [float(value) for value in values.mean(0)],
        "quantiles": {
            str(float(q)): [float(value) for value in torch.quantile(
                values, q, dim=0)] for q in quantiles},
        "columns": [
            "density_distance", "route_confidence",
            "rule_confidence", "candidate_confidence"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-checkpoint", type=Path, required=True)
    parser.add_argument("--consolidator-checkpoint", type=Path, required=True)
    parser.add_argument("--pairwise-transfer-checkpoint", type=Path, required=True)
    parser.add_argument("--projection-transfer-checkpoint", type=Path, required=True)
    parser.add_argument("--transfer-strength", type=float, default=.01)
    parser.add_argument("--lifetimes", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=521)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    transfers = (
        str(args.pairwise_transfer_checkpoint),
        str(args.projection_transfer_checkpoint))
    model, consolidator = _load(
        args.controller_checkpoint, args.consolidator_checkpoint, device,
        transfer_paths=transfers, transfer_strength=args.transfer_strength)
    result = {"schema": "factorized-router-ood-probe-v1"}
    for index, (primitive, generator) in enumerate(GENERATORS.items()):
        values = _collect(
            model, consolidator, generator,
            start=31_000_000 + index * 1_000_000,
            lifetimes=args.lifetimes, batch_size=args.batch_size,
            device=device)
        result[primitive] = _summary(values)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

"""Discarded probe for the capacity of generic memory-read statistics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from .model import UnifiedCognitiveController
from .train import seed_everything
from .train_adaptive_memory_read import _batch


@torch.no_grad()
def _metrics(
        probe: nn.Module, features: torch.Tensor,
        stored: torch.Tensor) -> dict[str, float]:
    probability = torch.sigmoid(probe(features).squeeze(-1))
    accepted = probability >= 0.5
    return {
        "accuracy": float((accepted == stored).float().mean()),
        "stored_accept_rate": float(accepted[stored].float().mean()),
        "absent_false_accept_rate": float(
            accepted[~stored].float().mean()),
        "probability_mean": float(probability.mean()),
    }


def _fit(
        probe: nn.Module, train_features: torch.Tensor,
        train_stored: torch.Tensor, test_features: torch.Tensor,
        test_stored: torch.Tensor, *, steps: int,
        learning_rate: float) -> dict[str, object]:
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
    target = train_stored.to(train_features.dtype)
    for _ in range(steps):
        logits = probe(train_features).squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return {
        "parameter_count": sum(
            parameter.numel() for parameter in probe.parameters()),
        "train": _metrics(probe, train_features, train_stored),
        "heldout": _metrics(probe, test_features, test_stored),
        "weights_discarded": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=5950)
    parser.add_argument("--contexts", type=int, default=8192)
    parser.add_argument("--bank-capacity", type=int, default=8)
    parser.add_argument("--write-threshold", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()
    if args.contexts % args.bank_capacity:
        raise ValueError("contexts must divide into complete memory banks")

    seed_everything(args.seed)
    device = torch.device(args.device)
    payload = torch.load(
        args.checkpoint, map_location=device, weights_only=False)
    configuration = dict(payload["model_configuration"])
    configuration["adaptive_memory_read"] = False
    model = UnifiedCognitiveController(**configuration).to(device)
    state_dict = {
        name: value for name, value in payload["state_dict"].items()
        if not name.startswith("memory_read_gate.")}
    model.load_state_dict(state_dict)
    model.eval()
    _, _, train_features, train_stored = _batch(
        model, count=args.contexts, capacity=args.bank_capacity,
        seed=args.seed + 1_000_000, device=device,
        write_threshold=args.write_threshold)
    _, _, test_features, test_stored = _batch(
        model, count=args.contexts, capacity=args.bank_capacity,
        seed=args.seed + 2_000_000, device=device,
        write_threshold=args.write_threshold)
    feature_names = [
        "cosine_match", "top_two_rank_margin",
        "selected_row_strength", "bank_occupancy"]
    feature_summary = {}
    for index, name in enumerate(feature_names):
        feature_summary[name] = {
            "stored_mean": float(
                test_features[test_stored, index].mean()),
            "absent_mean": float(
                test_features[~test_stored, index].mean()),
        }
    linear = nn.Linear(4, 1).to(device)
    nonlinear = nn.Sequential(
        nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 1)).to(device)
    report = {
        "schema": "unified-controller-memory-read-feature-probe-v1",
        "checkpoint": str(args.checkpoint),
        "seed": args.seed,
        "contexts_per_split": args.contexts,
        "bank_capacity": args.bank_capacity,
        "diagnostic_only": True,
        "private_stored_row_labels_used": True,
        "probe_weights_enter_agent": False,
        "feature_summary": feature_summary,
        "linear": _fit(
            linear, train_features, train_stored,
            test_features, test_stored, steps=args.steps,
            learning_rate=0.03),
        "nonlinear": _fit(
            nonlinear, train_features, train_stored,
            test_features, test_stored, steps=args.steps,
            learning_rate=0.01),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(
        report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

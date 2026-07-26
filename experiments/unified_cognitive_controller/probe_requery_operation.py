"""Measure whether a second-ranked latent memory re-query has decision value."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .audit_selective_disk import _query_keys, _support
from .environment import generate_lifetimes
from .probe_persistent_interface import _add_context_signatures
from .train import seed_everything
from .train_adaptive_memory_read import _outcomes
from .train_redundancy_transfer import build_transfer_arms


@torch.no_grad()
def requery_batch(
        model, *, count: int, capacity: int, seed: int,
        device: torch.device, write_threshold: float,
        ) -> tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return generic evidence, first-read outcome, and alternative outcome."""
    batch = _add_context_signatures(
        generate_lifetimes(
            count, 3, seed=seed, heldout=True,
            task="binary_mapping", support_trials=1, device=device),
        seed=seed + 10_000_000)
    keys, values, strengths = _support(model, batch, device=device)
    queries = _query_keys(model, batch, device=device)
    groups = count // capacity
    key_group = torch.nn.functional.normalize(
        keys.reshape(groups, capacity, -1), dim=-1)
    query_group = torch.nn.functional.normalize(
        queries.reshape(groups, capacity, -1), dim=-1)
    value_group = values.reshape(groups, capacity, -1)
    strength_group = strengths.reshape(groups, capacity)
    valid = strength_group >= write_threshold
    cosine = torch.einsum("gcw,gkw->gck", query_group, key_group)
    ranked = (
        cosine
        + strength_group.clamp_min(1e-6).log().unsqueeze(1))
    ranked = ranked.masked_fill(~valid.unsqueeze(1), -1e9)
    scores, selected = ranked.topk(2, dim=-1)
    gather_shape = (-1, -1, value_group.shape[-1])
    first = torch.gather(
        value_group, 1,
        selected[:, :, 0].unsqueeze(-1).expand(*gather_shape))
    second = torch.gather(
        value_group, 1,
        selected[:, :, 1].unsqueeze(-1).expand(*gather_shape))
    valid_count = valid.sum(-1, keepdim=True).expand(-1, capacity)
    first = torch.where(
        (valid_count >= 1).unsqueeze(-1), first,
        torch.zeros_like(first))
    second = torch.where(
        (valid_count >= 2).unsqueeze(-1), second,
        torch.zeros_like(second))
    first_index = selected[:, :, 0]
    confidence = torch.gather(
        cosine, 2, first_index.unsqueeze(-1)).squeeze(-1)
    margin = scores[:, :, 0] - scores[:, :, 1]
    margin = torch.where(
        valid_count == 1, torch.ones_like(margin), margin)
    usage = torch.gather(strength_group, 1, first_index)
    occupancy = (
        valid.to(values.dtype).sum(-1, keepdim=True) / capacity
    ).expand(-1, capacity).clone()
    empty = valid_count == 0
    confidence[empty] = 0
    margin[empty] = 0
    usage[empty] = 0
    occupancy[empty] = 0
    features = torch.stack(
        (confidence, margin, usage, occupancy), dim=-1)
    first_outcome = _outcomes(
        model, batch, first.reshape_as(values), device=device)
    second_outcome = _outcomes(
        model, batch, second.reshape_as(values), device=device)
    return (
        features.reshape(count, 4),
        first_outcome, second_outcome,
        (valid_count >= 2).reshape(-1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--selected-prefix", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=7881)
    parser.add_argument("--count", type=int, default=10240)
    parser.add_argument("--capacity", type=int, default=5)
    parser.add_argument("--write-threshold", type=float, default=0.5)
    parser.add_argument("--requery-cost", type=float, default=0.01)
    args = parser.parse_args()
    if args.count % args.capacity or args.count % 2:
        raise ValueError("count must be even and divide by capacity")

    seed_everything(args.seed)
    device = torch.device(args.device)
    parent = torch.load(
        args.parent_checkpoint, map_location=device, weights_only=False)
    selected = torch.load(
        args.selected_prefix, map_location=device, weights_only=False)
    controller = build_transfer_arms(
        parent, selected, device=device,
        fresh_seed=args.seed + 1)["selected_experience"]
    _, first, second, has_alternative = requery_batch(
        controller, count=args.count, capacity=args.capacity,
        seed=args.seed * 1_000_000, device=device,
        write_threshold=args.write_threshold)
    first_utility = first
    second_utility = second - args.requery_cost
    actual = torch.stack((first_utility, second_utility), dim=1)
    helps = second_utility > first_utility
    harms = second_utility < first_utility
    fixed = max(float(first_utility.mean()), float(second_utility.mean()))
    oracle = float(actual.max(-1).values.mean())
    report = {
        "schema": "second-ranked-requery-probe-v1",
        "configuration": {
            **vars(args),
            "parent_checkpoint": str(args.parent_checkpoint),
            "selected_prefix": str(args.selected_prefix),
            "report": str(args.report),
        },
        "training_performed": False,
        "learner_visible_verifier_bits": 0,
        "private_both_action_verifier_bits": args.count * 2,
        "alternative_available_fraction":
            float(has_alternative.float().mean()),
        "requery_helps_fraction": float(helps.float().mean()),
        "requery_harms_fraction": float(harms.float().mean()),
        "requery_neutral_fraction":
            float((~helps & ~harms).float().mean()),
        "always_first_utility": float(first_utility.mean()),
        "always_requery_utility": float(second_utility.mean()),
        "strongest_fixed_utility": fixed,
        "oracle_utility": oracle,
        "available_oracle_gap": oracle - fixed,
        "viable_for_adaptive_compute": (
            float(helps.float().mean()) >= 0.02
            and float(harms.float().mean()) >= 0.02
            and oracle >= fixed + 0.02),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Audit zero-shot reuse of a persisted latent skill on harder streams."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .probe_requery_operation import requery_batch
from .train import seed_everything
from .train_safe_requery_adaptation import (
    ActionValueHead,
    _load_head,
    head_from_skill_payload,
)
from .train_redundancy_transfer import build_transfer_arms
from .train_thought_compute_transfer import _metrics
from .verified_skill_store import VerifiedSkillStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--selected-prefix", type=Path, required=True)
    parser.add_argument("--parent-head-checkpoint", type=Path, required=True)
    parser.add_argument("--skill-store", type=Path, required=True)
    parser.add_argument("--skill-id", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=8030)
    parser.add_argument("--streams", type=int, default=8)
    parser.add_argument("--contexts", type=int, default=2016)
    parser.add_argument("--capacity", type=int, default=7)
    parser.add_argument("--write-threshold", type=float, default=0.5)
    parser.add_argument("--requery-cost", type=float, default=0.01)
    args = parser.parse_args()
    if args.contexts % args.capacity:
        raise ValueError("contexts must divide evenly by capacity")

    seed_everything(args.seed)
    device = torch.device(args.device)
    parent_payload = torch.load(
        args.parent_checkpoint, map_location=device, weights_only=False)
    selected_payload = torch.load(
        args.selected_prefix, map_location=device, weights_only=False)
    controller = build_transfer_arms(
        parent_payload, selected_payload, device=device,
        fresh_seed=args.seed + 1)["selected_experience"]
    parent = _load_head(args.parent_head_checkpoint, device)
    stored = VerifiedSkillStore(args.skill_store).load(
        args.skill_id, device=device)
    child = head_from_skill_payload(stored["payload"], device)
    if not isinstance(child, ActionValueHead):
        raise ValueError("reuse audit requires a stored action-value child")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed + 999)
        reset = ActionValueHead(
            child.network[1].out_features).to(device)

    streams = []
    for stream in range(args.streams):
        stream_seed = args.seed * 1_000_000 + stream
        features, first, second, _ = requery_batch(
            controller, count=args.contexts, capacity=args.capacity,
            seed=stream_seed, device=device,
            write_threshold=args.write_threshold)
        generator = torch.Generator(device=device).manual_seed(
            stream_seed + 50_000_000)
        permutation = torch.randperm(
            args.contexts, generator=generator, device=device)
        child_metrics = _metrics(
            child, features, first, second,
            thought_cost=args.requery_cost)
        shuffled_metrics = _metrics(
            child, features[permutation], first, second,
            thought_cost=args.requery_cost)
        with torch.no_grad():
            reversed_actions = child(features) <= 0
            reversed_utility = torch.where(
                reversed_actions, second - args.requery_cost, first).mean()
        streams.append({
            "stream": stream,
            "seed": stream_seed,
            "parent": _metrics(
                parent, features, first, second,
                thought_cost=args.requery_cost),
            "stored_child": child_metrics,
            "reset": _metrics(
                reset, features, first, second,
                thought_cost=args.requery_cost),
            "feature_shuffled_child": shuffled_metrics,
            "action_reversed_child_utility": float(reversed_utility),
        })

    def mean(path: tuple[str, ...]) -> float:
        values = []
        for row in streams:
            value = row
            for key in path:
                value = value[key]
            values.append(float(value))
        return sum(values) / len(values)

    child_utility = mean(("stored_child", "verified_utility"))
    parent_utility = mean(("parent", "verified_utility"))
    reset_utility = mean(("reset", "verified_utility"))
    shuffled_utility = mean((
        "feature_shuffled_child", "verified_utility"))
    reversed_utility = mean(("action_reversed_child_utility",))
    gate = {
        "child_beats_parent_by_2_points":
            child_utility >= parent_utility + 0.02,
        "child_beats_reset_by_2_points":
            child_utility >= reset_utility + 0.02,
        "feature_shuffle_degrades_by_2_points":
            shuffled_utility <= child_utility - 0.02,
        "action_reversal_degrades_by_2_points":
            reversed_utility <= child_utility - 0.02,
        "every_stream_child_beats_parent": all(
            row["stored_child"]["verified_utility"]
            > row["parent"]["verified_utility"]
            for row in streams),
    }
    gate["accepted"] = all(gate.values())
    report = {
        "schema": "compounding-reuse-audit-v1",
        "configuration": {
            **vars(args),
            "parent_checkpoint": str(args.parent_checkpoint),
            "selected_prefix": str(args.selected_prefix),
            "parent_head_checkpoint": str(args.parent_head_checkpoint),
            "skill_store": str(args.skill_store),
            "report": str(args.report),
        },
        "training_performed": False,
        "new_task_verifier_bits_used_for_learning": 0,
        "private_audit_both_action_bits":
            args.streams * args.contexts * 2,
        "means": {
            "parent_utility": parent_utility,
            "stored_child_utility": child_utility,
            "reset_utility": reset_utility,
            "feature_shuffled_child_utility": shuffled_utility,
            "action_reversed_child_utility": reversed_utility,
        },
        "streams": streams,
        "gate": gate,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

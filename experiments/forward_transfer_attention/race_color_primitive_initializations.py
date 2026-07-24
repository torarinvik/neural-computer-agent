"""Shared-experience race for color-primitive and relation-head initializations.

This script selects candidates only. Capability is established separately by
`audit_color_primitive_compounding` with full controls and blind replication.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch

from .audit_color_primitive_compounding import (
    ATOM_TRAIN_START,
    RELATION_TRAIN_START,
    TEST_START,
    _balanced_data,
    _evaluate_relation,
    _logged_outcomes,
    _passes_behavior,
)
from .audit_identify_near_transfer import (
    _balanced_indices,
    _subset_data,
    private_label,
    transfer_features,
)
from .audit_immutable_core_cross_primitive import _load_arms
from .train import seed_everything
from .train_identify_then_act import (
    fit_readout,
    identify_batch,
    make_readout,
)


def _fit_atom(
        features: torch.Tensor, labels: torch.Tensor, *,
        outcome_seed: int, initialization_seed: int,
        optimization_seed: int, updates: int,
        batch_size: int) -> torch.nn.Module:
    attempted, rewards, order = _logged_outcomes(
        labels, seed=outcome_seed)
    seed_everything(initialization_seed)
    initial = copy.deepcopy(make_readout(
        "antisymmetric", features.shape[-1], 64).state_dict())
    return fit_readout(
        initial, features[order.to(features.device)],
        attempted, rewards,
        readout_kind="antisymmetric", intention_width=64,
        updates=updates, batch_size=batch_size,
        learning_rate=3e-3, seed=optimization_seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--clones", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1901)
    parser.add_argument("--target-bits", type=int, default=64)
    parser.add_argument("--effect-bits", type=int, default=24)
    parser.add_argument("--relation-bits", type=int, default=64)
    parser.add_argument("--test-lifetimes", type=int, default=256)
    parser.add_argument("--atom-updates", type=int, default=128)
    parser.add_argument("--relation-updates", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    started = time.perf_counter()
    device = torch.device(args.device)
    arms = _load_arms(args.checkpoint, device=device)
    inherited = arms["immutable_core263"]
    fresh = arms["matched_fresh263"]

    target_data = _balanced_data(
        ATOM_TRAIN_START, args.target_bits,
        task="target_side", heldout=False, seed=args.seed + 10,
        relation_axis="color_salient")
    effect_data = _balanced_data(
        ATOM_TRAIN_START + 1_000_000, args.effect_bits,
        task="effect_side", heldout=False, seed=args.seed + 20,
        relation_axis="color_salient")
    relation_train = _balanced_data(
        RELATION_TRAIN_START, args.relation_bits,
        task="effect_target_match", heldout=False, seed=args.seed + 50,
        relation_axis="color_salient")
    test_pool = identify_batch(
        TEST_START, args.test_lifetimes * 2,
        heldout=True, relation_axis="color_salient")
    test_indices = _balanced_indices(
        private_label(test_pool, "effect_target_match"),
        args.test_lifetimes, seed=args.seed + 60)
    normal = _subset_data(test_pool, test_indices)
    datasets = {
        "train": relation_train,
        "normal": normal,
        "protocol_counterfactual": _subset_data(identify_batch(
            TEST_START, args.test_lifetimes * 2, heldout=True,
            relation_axis="color_salient",
            swap_protocol=True), test_indices),
        "target_counterfactual": _subset_data(identify_batch(
            TEST_START, args.test_lifetimes * 2, heldout=True,
            relation_axis="color_salient",
            reverse_target=True), test_indices),
        "missing_consequence": _subset_data(identify_batch(
            TEST_START, args.test_lifetimes * 2, heldout=True,
            relation_axis="color_salient",
            missing_consequence=True), test_indices),
        "missing_target": _subset_data(identify_batch(
            TEST_START, args.test_lifetimes * 2, heldout=True,
            relation_axis="color_salient",
            missing_target=True), test_indices),
    }
    labels = {
        name: private_label(data, "effect_target_match")
        for name, data in datasets.items()
    }
    with torch.no_grad():
        target_atom_features = transfer_features(
            fresh, target_data, interface="event_vision", device=device)
        effect_atom_features = transfer_features(
            inherited, effect_data, interface="event_vision", device=device)
        target_features = {
            name: transfer_features(
                fresh, data, interface="event_vision", device=device)
            for name, data in datasets.items()
        }
        effect_features = {
            name: transfer_features(
                inherited, data, interface="event_vision", device=device)
            for name, data in datasets.items()
        }
    effect_initialization_seed = args.seed + 42
    effect_head = _fit_atom(
        effect_atom_features, private_label(effect_data, "effect_side"),
        outcome_seed=args.seed + 40,
        initialization_seed=effect_initialization_seed,
        optimization_seed=args.seed + 43,
        updates=args.atom_updates, batch_size=args.batch_size)
    with torch.no_grad():
        effect_latents = {
            name: effect_head(features)
            for name, features in effect_features.items()
        }
    attempted, rewards, order = _logged_outcomes(
        labels["train"], seed=args.seed + 70)
    prefixes = tuple(
        value for value in (16, 24, 32, 64)
        if value <= args.relation_bits)
    candidates = []
    for clone in range(args.clones):
        target_initialization_seed = args.seed + 32 + clone * 101
        relation_initialization_seed = args.seed + 80 + clone * 103
        target_head = _fit_atom(
            target_atom_features,
            private_label(target_data, "target_side"),
            outcome_seed=args.seed + 30,
            initialization_seed=target_initialization_seed,
            optimization_seed=args.seed + 33,
            updates=args.atom_updates, batch_size=args.batch_size)
        with torch.no_grad():
            split_features = {
                name: torch.cat([
                    target_head(target_features[name]),
                    effect_latents[name],
                ], dim=-1)
                for name in datasets
            }
        seed_everything(relation_initialization_seed)
        initial_relation = copy.deepcopy(make_readout(
            "antisymmetric", 4, 16).state_dict())
        curve = []
        for prefix in prefixes:
            model = fit_readout(
                initial_relation,
                split_features["train"][order.to(device)][:prefix],
                attempted[:prefix], rewards[:prefix],
                readout_kind="antisymmetric", intention_width=16,
                updates=args.relation_updates,
                batch_size=args.batch_size,
                learning_rate=3e-3, seed=args.seed + 100 + prefix)
            metrics = _evaluate_relation(model, split_features, labels)
            causal_floor = min(
                float(metrics["normal_accuracy"]),
                float(metrics["protocol_counterfactual_accuracy"]),
                float(metrics["target_counterfactual_accuracy"]),
                float(metrics["protocol_counterfactual_flip_rate"]),
                float(metrics["target_counterfactual_flip_rate"]),
            )
            curve.append({
                "unique_relation_reward_bits": prefix,
                "causal_floor": causal_floor,
                "passes_behavior_gates": _passes_behavior(metrics),
                **metrics,
            })
        stable = next((
            point["unique_relation_reward_bits"]
            for index, point in enumerate(curve)
            if all(later["passes_behavior_gates"]
                   for later in curve[index:])
        ), None)
        candidates.append({
            "clone": clone,
            "target_initialization_seed": target_initialization_seed,
            "effect_initialization_seed": effect_initialization_seed,
            "relation_initialization_seed": relation_initialization_seed,
            "stable_relation_bits": stable,
            "causal_floor_sum": sum(
                point["causal_floor"] for point in curve),
            "curve": curve,
        })
        print(json.dumps({
            "clone": clone, "stable": stable,
            "causal_floor_sum": candidates[-1]["causal_floor_sum"],
        }, sort_keys=True), flush=True)
    ranked = sorted(candidates, key=lambda candidate: (
        candidate["stable_relation_bits"] is None,
        candidate["stable_relation_bits"]
        if candidate["stable_relation_bits"] is not None else 10**9,
        -candidate["causal_floor_sum"],
        candidate["clone"],
    ))
    report = {
        "schema": "color-primitive-initialization-race-v1",
        "selection_only_no_capability_claim": True,
        "configuration": vars(args) | {
            "checkpoint": str(args.checkpoint),
            "report": str(args.report),
        },
        "experience_accounting": {
            "target_atom_unique_reward_bits": args.target_bits,
            "effect_atom_unique_reward_bits": args.effect_bits,
            "relation_unique_reward_bits": args.relation_bits,
            "same_outcome_stream_for_every_clone": True,
            "environment_experience_not_multiplied_by_clones": True,
            "search_compute_is_multiplied_by_clones": True,
        },
        "candidates": candidates,
        "selected_parent": ranked[0],
        "passing_clone_count": sum(
            candidate["stable_relation_bits"] is not None
            for candidate in candidates),
        "total_seconds": time.perf_counter() - started,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(
        report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "selected_parent": report["selected_parent"],
        "passing_clone_count": report["passing_clone_count"],
        "total_seconds": report["total_seconds"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

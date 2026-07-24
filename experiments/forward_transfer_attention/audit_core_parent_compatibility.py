"""Audit whether population-selected core 263 preserves earlier task surfaces."""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch

from .map_identify_ignition_interval import _clean, _initial_readout
from .race_variance_decomposition import state_is_bit_identical
from .train import seed_everything
from .train_identify_then_act import (
    ActionHistoryCore,
    PRETRAIN_START,
    TEST_START,
    TRAIN_START,
    decision_features,
    evaluate,
    fit_readout,
    identify_batch,
    predictive_metrics,
    pretrain_core,
)


CORE_SEEDS = (211, 263)
TASKS = {
    "fixed_probe": {
        "prefixes": (8, 16, 32),
        "batch": {"fixed_probe_action": 0},
        "target_reversal_required": True,
        "established_threshold": 16,
    },
    "fixed_target": {
        "prefixes": (16, 32, 48, 64),
        "batch": {"fixed_target_direction": -1},
        "target_reversal_required": False,
        "established_threshold": 64,
    },
}


def task_gate(
        audit: dict[str, float], controls: dict[str, dict[str, float]], *,
        target_reversal_required: bool) -> bool:
    return bool(
        audit["normal_accuracy"] >= 0.75 and
        audit["protocol_swap_accuracy"] >= 0.75 and
        audit["protocol_swap_prediction_flip"] >= 0.75 and
        (not target_reversal_required or (
            audit["target_reverse_accuracy"] >= 0.75 and
            audit["target_reverse_prediction_flip"] >= 0.75)) and
        audit["missing_consequence_accuracy"] <= 0.60 and
        audit["missing_consequence_entropy"] > audit["normal_entropy"] and
        audit["no_probe_effect_accuracy"] <= 0.60 and
        controls["action_shuffled"]["verified_accuracy"] <= 0.60 and
        controls["reward_shuffled"]["verified_accuracy"] <= 0.60 and
        controls["fully_fresh_core"]["verified_accuracy"] <= 0.60)


def _fit(
        initial: dict[str, torch.Tensor], features: torch.Tensor,
        actions: torch.Tensor, rewards: torch.Tensor, *,
        prefix: int, updates: int, batch_size: int) -> torch.nn.Module:
    return fit_readout(
        initial, features[:prefix], actions[:prefix], rewards[:prefix],
        readout_kind="bottleneck", intention_width=64,
        updates=updates, batch_size=batch_size, learning_rate=3e-3,
        seed=211 + 80 + prefix)


def _audit(
        model: torch.nn.Module,
        features: dict[str, torch.Tensor],
        data: dict[str, dict[str, torch.Tensor]]) -> dict[str, float]:
    results = {
        name: evaluate(model, value, data[name]["correct_actions"])
        for name, value in features.items()
    }
    normal = results["normal"]
    protocol = results["protocol_swap"]
    target = results["target_reverse"]
    return {
        "normal_accuracy": normal["verified_accuracy"],
        "normal_entropy": normal["normalized_policy_entropy"],
        "protocol_swap_accuracy": protocol["verified_accuracy"],
        "protocol_swap_prediction_flip": float(
            (normal["predictions"] !=
             protocol["predictions"]).float().mean()),
        "target_reverse_accuracy": target["verified_accuracy"],
        "target_reverse_prediction_flip": float(
            (normal["predictions"] !=
             target["predictions"]).float().mean()),
        "missing_consequence_accuracy": (
            results["missing"]["verified_accuracy"]),
        "missing_consequence_entropy": (
            results["missing"]["normalized_policy_entropy"]),
        "no_probe_effect_accuracy": (
            results["no_effect"]["verified_accuracy"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--pretrain-lifetimes", type=int, default=128)
    parser.add_argument("--pretrain-steps", type=int, default=40)
    parser.add_argument("--test-lifetimes", type=int, default=256)
    parser.add_argument("--fit-updates", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    started = time.perf_counter()
    device = torch.device(args.device)
    pretrain = identify_batch(
        PRETRAIN_START, args.pretrain_lifetimes, heldout=False)
    initial = _initial_readout(seed=211 + 61, device=device)

    cores = {}
    for seed in CORE_SEEDS:
        seed_everything(seed)
        core = ActionHistoryCore(64).to(device)
        fresh = copy.deepcopy(core)
        pretraining = pretrain_core(
            core, pretrain, mode="action_conditioned",
            steps=args.pretrain_steps, batch_size=args.batch_size,
            learning_rate=3e-4, seed=211, device=device)
        cores[str(seed)] = {
            "core": core,
            "fresh": fresh,
            "state": {
                key: value.detach().clone()
                for key, value in core.state_dict().items()},
            "pretraining": pretraining,
        }

    results = {}
    for task_name, task in TASKS.items():
        maximum = max(task["prefixes"])
        kwargs = task["batch"]
        train = identify_batch(
            TRAIN_START, maximum, heldout=False, **kwargs)
        audit_data = {
            "normal": identify_batch(
                TEST_START, args.test_lifetimes,
                heldout=True, **kwargs),
            "protocol_swap": identify_batch(
                TEST_START, args.test_lifetimes,
                heldout=True, swap_protocol=True, **kwargs),
            "target_reverse": identify_batch(
                TEST_START, args.test_lifetimes,
                heldout=True, reverse_target=True, **kwargs),
            "missing": identify_batch(
                TEST_START, args.test_lifetimes,
                heldout=True, missing_consequence=True, **kwargs),
            "no_effect": identify_batch(
                TEST_START, args.test_lifetimes,
                heldout=True, no_probe_effect=True, **kwargs),
        }
        order = torch.randperm(
            maximum,
            generator=torch.Generator().manual_seed(211 + 59))
        actions = train["attempted_actions"][order]
        rewards = train["rewards"][order]
        task_results = {}
        for seed, core_data in cores.items():
            core = core_data["core"]
            train_features = decision_features(
                core, train, passive=False,
                device=device)[order.to(device)]
            audit_features = {
                name: decision_features(
                    core, data, passive=False, device=device)
                for name, data in audit_data.items()
            }
            fresh_train = decision_features(
                core_data["fresh"], train, passive=False,
                device=device)[order.to(device)]
            fresh_audit = decision_features(
                core_data["fresh"], audit_data["normal"],
                passive=False, device=device)
            points = []
            for prefix in task["prefixes"]:
                model = _fit(
                    initial, train_features, actions, rewards,
                    prefix=prefix, updates=args.fit_updates,
                    batch_size=args.batch_size)
                action_control = _fit(
                    initial, train_features,
                    1 - actions[:prefix], rewards,
                    prefix=prefix, updates=args.fit_updates,
                    batch_size=args.batch_size)
                reward_control = _fit(
                    initial, train_features, actions,
                    1.0 - rewards[:prefix],
                    prefix=prefix, updates=args.fit_updates,
                    batch_size=args.batch_size)
                fresh_control = _fit(
                    initial, fresh_train, actions, rewards,
                    prefix=prefix, updates=args.fit_updates,
                    batch_size=args.batch_size)
                controls = {
                    "action_shuffled": _clean(evaluate(
                        action_control, audit_features["normal"],
                        audit_data["normal"]["correct_actions"])),
                    "reward_shuffled": _clean(evaluate(
                        reward_control, audit_features["normal"],
                        audit_data["normal"]["correct_actions"])),
                    "fully_fresh_core": _clean(evaluate(
                        fresh_control, fresh_audit,
                        audit_data["normal"]["correct_actions"])),
                }
                audit = _audit(model, audit_features, audit_data)
                passed = task_gate(
                    audit, controls,
                    target_reversal_required=(
                        task["target_reversal_required"]))
                points.append({
                    "unique_reward_bits": prefix,
                    "audit": audit,
                    "controls": controls,
                    "passes": passed,
                })
                print(json.dumps({
                    "task": task_name, "core_seed": int(seed),
                    "prefix": prefix,
                    "normal_accuracy": audit["normal_accuracy"],
                    "passes": passed,
                }, sort_keys=True), flush=True)
            stable = next((
                point["unique_reward_bits"]
                for index, point in enumerate(points)
                if all(later["passes"] for later in points[index:])
            ), None)
            task_results[seed] = {
                "points": points,
                "stable_bits_to_all_gates": stable,
                "aulc_above_chance": sum(
                    max(0.0, point["audit"]["normal_accuracy"] - 0.5)
                    for point in points) / len(points),
            }
        results[task_name] = task_results

    retention = {}
    for seed, core_data in cores.items():
        after = predictive_metrics(
            core_data["core"],
            identify_batch(
                TEST_START, args.test_lifetimes, heldout=True),
            passive=False, device=device)
        bit_identical = state_is_bit_identical(
            core_data["state"], core_data["core"].state_dict())
        retention[seed] = {
            "core_parameters_bit_identical": bit_identical,
            "heldout_predictive_metrics_after": after,
            "passes": bit_identical,
        }

    candidate_compatible = all(
        results[task]["263"]["stable_bits_to_all_gates"] is not None and
        results[task]["263"]["stable_bits_to_all_gates"] <=
        TASKS[task]["established_threshold"]
        for task in TASKS
    ) and retention["263"]["passes"]
    report = {
        "schema": "core-parent-compatibility-v1",
        "semantic_labels_used_for_training": False,
        "unattempted_action_labels_used_for_training": False,
        "candidate_core_seed": 263,
        "anchor_core_seed": 211,
        "negative_control_transform": (
            "exact binary complement; action 0<->1 or reward 0<->1"),
        "configuration": vars(args) | {"report": str(args.report)},
        "task_definitions": TASKS,
        "results": results,
        "frozen_core_retention": retention,
        "candidate_preserves_established_thresholds": candidate_compatible,
        "general_agent_checkpoint_promoted": False,
        "total_seconds": time.perf_counter() - started,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "candidate_preserves_established_thresholds": candidate_compatible,
        "total_seconds": report["total_seconds"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()

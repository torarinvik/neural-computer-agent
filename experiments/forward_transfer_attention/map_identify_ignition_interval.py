"""Map the missing 32-to-64 verifier-bit ignition interval.

One shared predictive core, one ordered stream of 64 unique lifetimes, and one
answer-path initialization are used throughout.  Each prefix is fitted
independently so optimizer history cannot masquerade as additional experience.
Every point receives the same held-out pixel-rerender and shuffled-outcome
audits.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch

from .train import seed_everything
from .train_feature_interface_tournament import BLIND_START
from .train_identify_then_act import (
    ActionHistoryCore,
    PRETRAIN_START,
    TEST_START,
    TRAIN_START,
    decision_features,
    evaluate,
    fit_readout,
    identify_batch,
    make_readout,
    pretrain_core,
)


PREFIXES = (32, 40, 48, 56, 64)


def _initial_readout(
        *, seed: int, device: torch.device) -> dict[str, torch.Tensor]:
    seed_everything(seed)
    model = make_readout("bottleneck", 192, 64).to(device)
    return copy.deepcopy(model.state_dict())


def _fit(
        initial: dict[str, torch.Tensor], features: torch.Tensor,
        attempted: torch.Tensor, rewards: torch.Tensor, *,
        updates: int, batch_size: int, seed: int) -> torch.nn.Module:
    return fit_readout(
        initial, features, attempted, rewards,
        readout_kind="bottleneck", intention_width=64,
        updates=updates, batch_size=batch_size,
        learning_rate=3e-3, seed=seed)


def _clean(result: dict[str, object]) -> dict[str, float]:
    return {
        key: value for key, value in result.items()
        if key != "predictions"
    }


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


def _passes(
        audit: dict[str, float],
        controls: dict[str, dict[str, float]]) -> bool:
    return bool(
        audit["normal_accuracy"] >= 0.75 and
        audit["protocol_swap_accuracy"] >= 0.75 and
        audit["protocol_swap_prediction_flip"] >= 0.75 and
        audit["target_reverse_accuracy"] >= 0.75 and
        audit["target_reverse_prediction_flip"] >= 0.75 and
        audit["missing_consequence_accuracy"] <= 0.60 and
        audit["missing_consequence_entropy"] > audit["normal_entropy"] and
        audit["no_probe_effect_accuracy"] <= 0.60 and
        controls["action_shuffled"]["verified_accuracy"] <= 0.60 and
        controls["reward_shuffled"]["verified_accuracy"] <= 0.60 and
        controls["fully_fresh_core"]["verified_accuracy"] <= 0.60)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--pretrain-lifetimes", type=int, default=128)
    parser.add_argument("--pretrain-steps", type=int, default=40)
    parser.add_argument("--test-lifetimes", type=int, default=256)
    parser.add_argument("--fit-updates", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    started = time.perf_counter()
    seed_everything(args.seed)
    device = torch.device(args.device)

    pretrain = identify_batch(
        PRETRAIN_START, args.pretrain_lifetimes, heldout=False)
    train = identify_batch(TRAIN_START, max(PREFIXES), heldout=False)
    audit_data = {
        "normal": identify_batch(
            BLIND_START, args.test_lifetimes, heldout=True),
        "protocol_swap": identify_batch(
            BLIND_START, args.test_lifetimes, heldout=True,
            swap_protocol=True),
        "target_reverse": identify_batch(
            BLIND_START, args.test_lifetimes, heldout=True,
            reverse_target=True),
        "missing": identify_batch(
            BLIND_START, args.test_lifetimes, heldout=True,
            missing_consequence=True),
        "no_effect": identify_batch(
            BLIND_START, args.test_lifetimes, heldout=True,
            no_probe_effect=True),
    }
    # Reserve the old selection range even though this map reports only the
    # genuinely blind range.
    assert BLIND_START > TEST_START + args.test_lifetimes

    core = ActionHistoryCore(64).to(device)
    initial_core_state = copy.deepcopy(core.state_dict())
    pretraining = pretrain_core(
        core, pretrain, mode="action_conditioned",
        steps=args.pretrain_steps, batch_size=args.batch_size,
        learning_rate=3e-4, seed=args.seed, device=device)
    train_features = decision_features(
        core, train, passive=False, device=device)
    audit_features = {
        name: decision_features(
            core, data, passive=False, device=device)
        for name, data in audit_data.items()
    }
    fresh = ActionHistoryCore(64).to(device)
    fresh.load_state_dict(initial_core_state)
    fresh_train_features = decision_features(
        fresh, train, passive=False, device=device)
    fresh_blind_features = decision_features(
        fresh, audit_data["normal"], passive=False, device=device)

    order = torch.randperm(
        max(PREFIXES),
        generator=torch.Generator().manual_seed(args.seed + 59))
    train_features = train_features[order.to(device)]
    fresh_train_features = fresh_train_features[order.to(device)]
    attempted = train["attempted_actions"][order]
    rewards = train["rewards"][order]
    # Match train_identify_then_act exactly so this is an interpolation of the
    # established baseline rather than a new optimization-seed experiment.
    initial = _initial_readout(seed=args.seed + 61, device=device)

    points = []
    for prefix in PREFIXES:
        permutation = torch.randperm(
            prefix,
            generator=torch.Generator().manual_seed(
                args.seed + 71 + prefix))
        model = _fit(
            initial, train_features[:prefix], attempted[:prefix],
            rewards[:prefix], updates=args.fit_updates,
            batch_size=args.batch_size, seed=args.seed + 80 + prefix)
        audit = _audit(model, audit_features, audit_data)
        action_control = _fit(
            initial, train_features[:prefix],
            attempted[:prefix][permutation], rewards[:prefix],
            updates=args.fit_updates, batch_size=args.batch_size,
            seed=args.seed + 80 + prefix)
        reward_control = _fit(
            initial, train_features[:prefix], attempted[:prefix],
            rewards[:prefix][permutation], updates=args.fit_updates,
            batch_size=args.batch_size, seed=args.seed + 80 + prefix)
        fresh_control = _fit(
            initial, fresh_train_features[:prefix], attempted[:prefix],
            rewards[:prefix], updates=args.fit_updates,
            batch_size=args.batch_size, seed=args.seed + 80 + prefix)
        controls = {
            "action_shuffled": _clean(evaluate(
                action_control, audit_features["normal"],
                audit_data["normal"]["correct_actions"])),
            "reward_shuffled": _clean(evaluate(
                reward_control, audit_features["normal"],
                audit_data["normal"]["correct_actions"])),
            "fully_fresh_core": _clean(evaluate(
                fresh_control, fresh_blind_features,
                audit_data["normal"]["correct_actions"])),
        }
        point = {
            "unique_reward_bits": prefix,
            "unique_lifetimes": prefix,
            "optimizer_updates": args.fit_updates,
            "examples_processed": (
                args.fit_updates * min(args.batch_size, prefix)),
            "audit": audit,
            "controls": controls,
            "passes_all_gates": _passes(audit, controls),
        }
        points.append(point)
        print(json.dumps(point, sort_keys=True), flush=True)

    stable = next((
        point["unique_reward_bits"]
        for index, point in enumerate(points)
        if all(later["passes_all_gates"] for later in points[index:])
    ), None)
    first = next((
        point["unique_reward_bits"] for point in points
        if point["passes_all_gates"]), None)
    report = {
        "schema": "identify-ignition-interval-v2",
        "seed_convention": (
            "exact-train-identify-v1: readout initialization seed+61; "
            "fit/minibatch seed+80+prefix; shuffle seed+71+prefix; "
            "fresh core shares the original untrained initialization"),
        "semantic_labels_used_for_training": False,
        "unattempted_action_labels_used_for_training": False,
        "same_core_and_readout_initialization_across_prefixes": True,
        "independent_fit_at_each_prefix": True,
        "unique_reward_bits_are_not_replay_examples": True,
        "configuration": vars(args) | {"report": str(args.report)},
        "pretraining": pretraining,
        "points": points,
        "first_passing_bits": first,
        "stable_passing_bits": stable,
        "total_seconds": time.perf_counter() - started,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "first_passing_bits": first,
        "stable_passing_bits": stable,
        "total_seconds": report["total_seconds"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()

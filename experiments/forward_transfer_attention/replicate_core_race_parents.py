"""Replicate the two causal horse-race parents on a disjoint experience stream."""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch

from .map_identify_ignition_interval import (
    _audit,
    _clean,
    _initial_readout,
    _passes,
)
from .race_variance_decomposition import (
    causal_floor,
    state_is_bit_identical,
)
from .train import seed_everything
from .train_identify_then_act import (
    ActionHistoryCore,
    PRETRAIN_START,
    TRAIN_START,
    decision_features,
    evaluate,
    fit_readout,
    identify_batch,
    predictive_metrics,
    pretrain_core,
)


PARENT_SEEDS = (211, 263)
PREFIXES = (32, 48, 64)
REPLICATION_SEED = 307
REPLICATION_TRAIN_START = TRAIN_START + 20_000_000
REPLICATION_AUDIT_START = TRAIN_START + 30_000_000


def _fit(
        initial: dict[str, torch.Tensor],
        features: torch.Tensor, attempted: torch.Tensor,
        rewards: torch.Tensor, *, prefix: int,
        updates: int, batch_size: int) -> torch.nn.Module:
    return fit_readout(
        initial, features[:prefix], attempted[:prefix],
        rewards[:prefix], readout_kind="bottleneck",
        intention_width=64, updates=updates,
        batch_size=batch_size, learning_rate=3e-3,
        seed=REPLICATION_SEED + 80 + prefix)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--pretrain-lifetimes", type=int, default=128)
    parser.add_argument("--pretrain-steps", type=int, default=40)
    parser.add_argument("--audit-lifetimes", type=int, default=256)
    parser.add_argument("--fit-updates", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    started = time.perf_counter()
    device = torch.device(args.device)

    pretrain = identify_batch(
        PRETRAIN_START, args.pretrain_lifetimes, heldout=False)
    train = identify_batch(
        REPLICATION_TRAIN_START, max(PREFIXES), heldout=False)
    audit_data = {
        "normal": identify_batch(
            REPLICATION_AUDIT_START, args.audit_lifetimes, heldout=True),
        "protocol_swap": identify_batch(
            REPLICATION_AUDIT_START, args.audit_lifetimes, heldout=True,
            swap_protocol=True),
        "target_reverse": identify_batch(
            REPLICATION_AUDIT_START, args.audit_lifetimes, heldout=True,
            reverse_target=True),
        "missing": identify_batch(
            REPLICATION_AUDIT_START, args.audit_lifetimes, heldout=True,
            missing_consequence=True),
        "no_effect": identify_batch(
            REPLICATION_AUDIT_START, args.audit_lifetimes, heldout=True,
            no_probe_effect=True),
    }
    order = torch.randperm(
        max(PREFIXES),
        generator=torch.Generator().manual_seed(REPLICATION_SEED + 59))
    attempted = train["attempted_actions"][order]
    rewards = train["rewards"][order]
    initial = _initial_readout(
        seed=REPLICATION_SEED + 61, device=device)

    parents = {}
    for parent_seed in PARENT_SEEDS:
        seed_everything(parent_seed)
        core = ActionHistoryCore(64).to(device)
        fresh = copy.deepcopy(core)
        pretraining = pretrain_core(
            core, pretrain, mode="action_conditioned",
            steps=args.pretrain_steps, batch_size=args.batch_size,
            learning_rate=3e-4, seed=211, device=device)
        state = {
            key: value.detach().clone()
            for key, value in core.state_dict().items()}
        before = predictive_metrics(
            core, audit_data["normal"],
            passive=False, device=device)
        train_features = decision_features(
            core, train, passive=False, device=device)[order.to(device)]
        audit_features = {
            name: decision_features(
                core, data, passive=False, device=device)
            for name, data in audit_data.items()
        }
        fresh_train = decision_features(
            fresh, train, passive=False, device=device)[order.to(device)]
        fresh_audit = decision_features(
            fresh, audit_data["normal"],
            passive=False, device=device)
        points, models = [], {}
        for prefix in PREFIXES:
            permutation = torch.randperm(
                prefix,
                generator=torch.Generator().manual_seed(
                    REPLICATION_SEED + 71 + prefix))
            model = _fit(
                initial, train_features, attempted, rewards,
                prefix=prefix, updates=args.fit_updates,
                batch_size=args.batch_size)
            action_control = _fit(
                initial, train_features,
                attempted[:prefix][permutation], rewards,
                prefix=prefix, updates=args.fit_updates,
                batch_size=args.batch_size)
            reward_control = _fit(
                initial, train_features, attempted,
                rewards[:prefix][permutation],
                prefix=prefix, updates=args.fit_updates,
                batch_size=args.batch_size)
            fresh_control = _fit(
                initial, fresh_train, attempted, rewards,
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
            point = {
                "unique_reward_bits": prefix,
                "optimizer_updates": args.fit_updates,
                "examples_processed": (
                    args.fit_updates * min(args.batch_size, prefix)),
                "audit": audit,
                "causal_floor": causal_floor(audit),
                "controls": controls,
                "passes_all_behavior_gates": _passes(audit, controls),
            }
            points.append(point)
            models[prefix] = model
            print(json.dumps({
                "parent_seed": parent_seed,
                "prefix": prefix,
                "normal_accuracy": audit["normal_accuracy"],
                "causal_floor": point["causal_floor"],
                "passes": point["passes_all_behavior_gates"],
            }, sort_keys=True), flush=True)
        after = predictive_metrics(
            core, audit_data["normal"],
            passive=False, device=device)
        retention = {
            "core_parameters_bit_identical": state_is_bit_identical(
                state, core.state_dict()),
            "predictive_metrics_before": before,
            "predictive_metrics_after": after,
            "passes": (
                state_is_bit_identical(state, core.state_dict()) and
                before == after),
        }
        stable = next((
            point["unique_reward_bits"]
            for index, point in enumerate(points)
            if all(
                later["passes_all_behavior_gates"]
                for later in points[index:])
        ), None)
        parents[str(parent_seed)] = {
            "parent_seed": parent_seed,
            "pretraining": pretraining,
            "points": points,
            "stable_bits_to_all_behavior_gates": stable,
            "retention": retention,
            "admitted_to_old_primitive_retention_suite": (
                stable is not None and retention["passes"]),
        }

    admitted = [
        value for value in parents.values()
        if value["admitted_to_old_primitive_retention_suite"]]
    winner = None
    if admitted:
        best = min(
            admitted,
            key=lambda value: (
                value["stable_bits_to_all_behavior_gates"],
                -sum(point["causal_floor"] for point in value["points"]),
            ))
        winner = str(best["parent_seed"])
    report = {
        "schema": "core-race-parent-replication-v1",
        "semantic_labels_used_for_training": False,
        "unattempted_action_labels_used_for_training": False,
        "disjoint_policy_experience_from_selection_race": True,
        "different_downstream_initialization_and_sampling": True,
        "unique_environment_reward_bits_generated": max(PREFIXES),
        "retention_is_an_eligibility_gate": True,
        "configuration": vars(args) | {"report": str(args.report)},
        "parent_seeds": list(PARENT_SEEDS),
        "replication_seed": REPLICATION_SEED,
        "replication_train_start": REPLICATION_TRAIN_START,
        "replication_audit_start": REPLICATION_AUDIT_START,
        "parents": parents,
        "winner_admitted_to_old_primitive_retention_suite": winner,
        "checkpoint_promoted": False,
        "total_seconds": time.perf_counter() - started,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "winner_admitted_to_old_primitive_retention_suite": winner,
        "total_seconds": report["total_seconds"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()

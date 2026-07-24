"""Causal horse race which localizes learning-seed variance at 64 outcomes."""
from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .map_identify_ignition_interval import (
    _audit,
    _clean,
    _initial_readout,
    _passes,
)
from .train import seed_everything
from .train_feature_interface_tournament import BLIND_START
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


ANCHOR_SEED = 211
LEVELS = (151, 211, 307)
UNIQUE_REWARD_BITS = 64


@dataclass(frozen=True)
class Horse:
    name: str
    factor: str
    level: int
    core_init_seed: int = ANCHOR_SEED
    pretrain_sampling_seed: int = ANCHOR_SEED
    readout_init_seed: int = ANCHOR_SEED + 61
    readout_sampling_seed: int = ANCHOR_SEED + 80 + UNIQUE_REWARD_BITS


HORSES = (
    Horse("anchor", "anchor", ANCHOR_SEED),
    Horse(
        "core_init_151", "core_initialization", 151,
        core_init_seed=151),
    Horse(
        "core_init_307", "core_initialization", 307,
        core_init_seed=307),
    Horse(
        "pretrain_sampling_151", "pretrain_sampling", 151,
        pretrain_sampling_seed=151),
    Horse(
        "pretrain_sampling_307", "pretrain_sampling", 307,
        pretrain_sampling_seed=307),
    Horse(
        "readout_init_151", "readout_initialization", 151,
        readout_init_seed=151 + 61),
    Horse(
        "readout_init_307", "readout_initialization", 307,
        readout_init_seed=307 + 61),
    Horse(
        "readout_sampling_151", "readout_sampling", 151,
        readout_sampling_seed=151 + 80 + UNIQUE_REWARD_BITS),
    Horse(
        "readout_sampling_307", "readout_sampling", 307,
        readout_sampling_seed=307 + 80 + UNIQUE_REWARD_BITS),
)


def causal_floor(audit: dict[str, float]) -> float:
    return min(
        audit["normal_accuracy"],
        audit["protocol_swap_accuracy"],
        audit["protocol_swap_prediction_flip"],
        audit["target_reverse_accuracy"],
        audit["target_reverse_prediction_flip"],
    )


def state_is_bit_identical(
        before: dict[str, torch.Tensor],
        after: dict[str, torch.Tensor]) -> bool:
    return before.keys() == after.keys() and all(
        torch.equal(before[key], after[key]) for key in before)


def _make_core(
        horse: Horse, pretrain_data: dict[str, torch.Tensor], *,
        pretrain_steps: int, batch_size: int,
        device: torch.device) -> tuple[
            ActionHistoryCore, ActionHistoryCore, dict[str, object]]:
    seed_everything(horse.core_init_seed)
    core = ActionHistoryCore(64).to(device)
    fresh = copy.deepcopy(core)
    training = pretrain_core(
        core, pretrain_data, mode="action_conditioned",
        steps=pretrain_steps, batch_size=batch_size,
        learning_rate=3e-4, seed=horse.pretrain_sampling_seed,
        device=device)
    return core, fresh, training


def _fit(
        initial: dict[str, torch.Tensor],
        features: torch.Tensor, actions: torch.Tensor,
        rewards: torch.Tensor, *, updates: int,
        batch_size: int, seed: int) -> torch.nn.Module:
    return fit_readout(
        initial, features, actions, rewards,
        readout_kind="bottleneck", intention_width=64,
        updates=updates, batch_size=batch_size,
        learning_rate=3e-3, seed=seed)


def _factor_ranges(results: dict[str, dict]) -> dict[str, dict]:
    anchor = results["anchor"]
    ranges = {}
    for factor in (
            "core_initialization", "pretrain_sampling",
            "readout_initialization", "readout_sampling"):
        members = [
            anchor,
            *(value for value in results.values()
              if value["configuration"]["factor"] == factor),
        ]
        normal = [
            value["audit"]["normal_accuracy"] for value in members]
        floors = [value["causal_floor"] for value in members]
        ranges[factor] = {
            "normal_accuracy_range": max(normal) - min(normal),
            "causal_floor_range": max(floors) - min(floors),
            "levels": [value["configuration"]["level"] for value in members],
        }
    return ranges


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
    train = identify_batch(
        TRAIN_START, UNIQUE_REWARD_BITS, heldout=False)
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
    order = torch.randperm(
        UNIQUE_REWARD_BITS,
        generator=torch.Generator().manual_seed(ANCHOR_SEED + 59))
    attempted = train["attempted_actions"][order]
    rewards = train["rewards"][order]
    permutation = torch.randperm(
        UNIQUE_REWARD_BITS,
        generator=torch.Generator().manual_seed(
            ANCHOR_SEED + 71 + UNIQUE_REWARD_BITS))

    core_cache = {}
    results = {}
    for horse in HORSES:
        core_key = (
            horse.core_init_seed, horse.pretrain_sampling_seed)
        if core_key not in core_cache:
            core, fresh, pretraining = _make_core(
                horse, pretrain,
                pretrain_steps=args.pretrain_steps,
                batch_size=args.batch_size, device=device)
            train_features = decision_features(
                core, train, passive=False, device=device)[order.to(device)]
            audit_features = {
                name: decision_features(
                    core, data, passive=False, device=device)
                for name, data in audit_data.items()
            }
            fresh_train = decision_features(
                fresh, train, passive=False, device=device)[order.to(device)]
            fresh_blind = decision_features(
                fresh, audit_data["normal"],
                passive=False, device=device)
            predictive = predictive_metrics(
                core, audit_data["normal"],
                passive=False, device=device)
            core_cache[core_key] = {
                "core": core,
                "fresh": fresh,
                "pretraining": pretraining,
                "train_features": train_features,
                "audit_features": audit_features,
                "fresh_train": fresh_train,
                "fresh_blind": fresh_blind,
                "predictive_before": predictive,
            }
        cached = core_cache[core_key]
        core = cached["core"]
        state_before = {
            key: value.detach().clone()
            for key, value in core.state_dict().items()}
        initial = _initial_readout(
            seed=horse.readout_init_seed, device=device)
        model = _fit(
            initial, cached["train_features"], attempted, rewards,
            updates=args.fit_updates, batch_size=args.batch_size,
            seed=horse.readout_sampling_seed)
        action_control = _fit(
            initial, cached["train_features"],
            attempted[permutation], rewards,
            updates=args.fit_updates, batch_size=args.batch_size,
            seed=horse.readout_sampling_seed)
        reward_control = _fit(
            initial, cached["train_features"], attempted,
            rewards[permutation], updates=args.fit_updates,
            batch_size=args.batch_size,
            seed=horse.readout_sampling_seed)
        fresh_control = _fit(
            initial, cached["fresh_train"], attempted, rewards,
            updates=args.fit_updates, batch_size=args.batch_size,
            seed=horse.readout_sampling_seed)
        controls = {
            "action_shuffled": _clean(evaluate(
                action_control, cached["audit_features"]["normal"],
                audit_data["normal"]["correct_actions"])),
            "reward_shuffled": _clean(evaluate(
                reward_control, cached["audit_features"]["normal"],
                audit_data["normal"]["correct_actions"])),
            "fully_fresh_core": _clean(evaluate(
                fresh_control, cached["fresh_blind"],
                audit_data["normal"]["correct_actions"])),
        }
        audit = _audit(
            model, cached["audit_features"], audit_data)
        predictive_after = predictive_metrics(
            core, audit_data["normal"],
            passive=False, device=device)
        retention = {
            "core_parameters_bit_identical": state_is_bit_identical(
                state_before, core.state_dict()),
            "predictive_metrics_before": cached["predictive_before"],
            "predictive_metrics_after": predictive_after,
            "heldout_predictive_loss_change": (
                predictive_after["heldout_standardized_loss"] -
                cached["predictive_before"]["heldout_standardized_loss"]),
            "passes": (
                state_is_bit_identical(state_before, core.state_dict()) and
                predictive_after == cached["predictive_before"]),
        }
        eligible = _passes(audit, controls) and retention["passes"]
        result = {
            "configuration": asdict(horse),
            "unique_reward_bits": UNIQUE_REWARD_BITS,
            "optimizer_updates": args.fit_updates,
            "examples_processed": (
                args.fit_updates *
                min(args.batch_size, UNIQUE_REWARD_BITS)),
            "audit": audit,
            "causal_floor": causal_floor(audit),
            "controls": controls,
            "retention": retention,
            "eligible": eligible,
        }
        results[horse.name] = result
        print(json.dumps({
            "horse": horse.name,
            "normal_accuracy": audit["normal_accuracy"],
            "causal_floor": result["causal_floor"],
            "retention_passes": retention["passes"],
            "eligible": eligible,
        }, sort_keys=True), flush=True)

    factor_ranges = _factor_ranges(results)
    dominant = max(
        factor_ranges,
        key=lambda factor: factor_ranges[factor]["causal_floor_range"])
    eligible = [
        name for name, value in results.items() if value["eligible"]]
    winner = (
        max(eligible, key=lambda name: results[name]["causal_floor"])
        if eligible else None)
    report = {
        "schema": "causal-horse-race-variance-v1",
        "semantic_labels_used_for_training": False,
        "unattempted_action_labels_used_for_training": False,
        "same_sensory_experience_for_every_horse": True,
        "unique_reward_bits_per_horse": UNIQUE_REWARD_BITS,
        "unique_environment_reward_bits_generated": UNIQUE_REWARD_BITS,
        "search_reward_bit_presentations": (
            UNIQUE_REWARD_BITS * len(HORSES)),
        "horse_count": len(HORSES),
        "retention_is_an_eligibility_gate": True,
        "configuration": vars(args) | {"report": str(args.report)},
        "horses": results,
        "factor_ranges": factor_ranges,
        "dominant_variance_factor_by_causal_floor": dominant,
        "eligible_horses": eligible,
        "winner": winner,
        "pretraining_by_core": {
            f"init_{key[0]}_sampling_{key[1]}": value["pretraining"]
            for key, value in core_cache.items()
        },
        "total_seconds": time.perf_counter() - started,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "dominant_variance_factor": dominant,
        "eligible_horses": eligible,
        "winner": winner,
        "total_seconds": report["total_seconds"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()

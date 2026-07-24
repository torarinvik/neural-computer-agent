"""Successive-halving horse race over predictive-core initializations."""
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
    ANCHOR_SEED,
    causal_floor,
    state_is_bit_identical,
)
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
    predictive_metrics,
    pretrain_core,
)


CORE_SEEDS = (43, 97, 211, 263, 401, 503)
STAGES = (32, 48, 64)


def select_survivors(
        names: list[str], records: dict[str, dict], *,
        count: int, stage: int) -> list[str]:
    """Keep strong horses plus a distinct late-ignition candidate."""
    if count >= len(names):
        return list(names)
    ordered = sorted(
        names, key=lambda name: records[name]["causal_floor"],
        reverse=True)
    if stage == 32:
        chosen = ordered[:count - 1]
        remaining = [name for name in names if name not in chosen]
        chosen.append(max(
            remaining,
            key=lambda name: records[name]["mechanistic_score"]))
        return chosen

    # At 48 bits preserve one current leader, one fastest improver, and one
    # reward-free mechanistic candidate.  Fill any duplicate slots by score.
    chosen = [ordered[0]]
    remaining = [name for name in names if name not in chosen]
    if len(chosen) < count:
        progress = max(
            remaining, key=lambda name: records[name]["progress"])
        chosen.append(progress)
        remaining.remove(progress)
    if len(chosen) < count and remaining:
        mechanistic = max(
            remaining,
            key=lambda name: records[name]["mechanistic_score"])
        chosen.append(mechanistic)
        remaining.remove(mechanistic)
    for name in ordered:
        if len(chosen) >= count:
            break
        if name not in chosen:
            chosen.append(name)
    return chosen


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
        seed=ANCHOR_SEED + 80 + prefix)


def _stage_audit(
        model: torch.nn.Module,
        features: dict[str, torch.Tensor],
        data: dict[str, dict[str, torch.Tensor]]) -> dict[str, float]:
    results = {
        name: evaluate(
            model, features[name], data[name]["correct_actions"])
        for name in ("normal", "protocol_swap", "target_reverse")
    }
    normal = results["normal"]
    protocol = results["protocol_swap"]
    target = results["target_reverse"]
    return {
        "normal_accuracy": normal["verified_accuracy"],
        "protocol_swap_accuracy": protocol["verified_accuracy"],
        "protocol_swap_prediction_flip": float(
            (normal["predictions"] !=
             protocol["predictions"]).float().mean()),
        "target_reverse_accuracy": target["verified_accuracy"],
        "target_reverse_prediction_flip": float(
            (normal["predictions"] !=
             target["predictions"]).float().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--pretrain-lifetimes", type=int, default=128)
    parser.add_argument("--pretrain-steps", type=int, default=40)
    parser.add_argument("--selection-lifetimes", type=int, default=256)
    parser.add_argument("--blind-lifetimes", type=int, default=256)
    parser.add_argument("--fit-updates", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    started = time.perf_counter()
    device = torch.device(args.device)

    pretrain = identify_batch(
        PRETRAIN_START, args.pretrain_lifetimes, heldout=False)
    train = identify_batch(TRAIN_START, max(STAGES), heldout=False)
    selection_data = {
        "normal": identify_batch(
            TEST_START, args.selection_lifetimes, heldout=True),
        "protocol_swap": identify_batch(
            TEST_START, args.selection_lifetimes, heldout=True,
            swap_protocol=True),
        "target_reverse": identify_batch(
            TEST_START, args.selection_lifetimes, heldout=True,
            reverse_target=True),
    }
    blind_data = {
        "normal": identify_batch(
            BLIND_START, args.blind_lifetimes, heldout=True),
        "protocol_swap": identify_batch(
            BLIND_START, args.blind_lifetimes, heldout=True,
            swap_protocol=True),
        "target_reverse": identify_batch(
            BLIND_START, args.blind_lifetimes, heldout=True,
            reverse_target=True),
        "missing": identify_batch(
            BLIND_START, args.blind_lifetimes, heldout=True,
            missing_consequence=True),
        "no_effect": identify_batch(
            BLIND_START, args.blind_lifetimes, heldout=True,
            no_probe_effect=True),
    }
    order = torch.randperm(
        max(STAGES),
        generator=torch.Generator().manual_seed(ANCHOR_SEED + 59))
    attempted = train["attempted_actions"][order]
    rewards = train["rewards"][order]
    control_permutation = torch.randperm(
        max(STAGES),
        generator=torch.Generator().manual_seed(
            ANCHOR_SEED + 71 + max(STAGES)))
    initial = _initial_readout(
        seed=ANCHOR_SEED + 61, device=device)

    horses = {}
    for core_seed in CORE_SEEDS:
        name = f"core_{core_seed}"
        seed_everything(core_seed)
        core = ActionHistoryCore(64).to(device)
        fresh = copy.deepcopy(core)
        pretraining = pretrain_core(
            core, pretrain, mode="action_conditioned",
            steps=args.pretrain_steps, batch_size=args.batch_size,
            learning_rate=3e-4, seed=ANCHOR_SEED, device=device)
        state = {
            key: value.detach().clone()
            for key, value in core.state_dict().items()}
        train_features = decision_features(
            core, train, passive=False, device=device)[order.to(device)]
        selection_features = {
            key: decision_features(
                core, data, passive=False, device=device)
            for key, data in selection_data.items()
        }
        mechanistic = predictive_metrics(
            core, selection_data["normal"],
            passive=False, device=device)
        horses[name] = {
            "core_seed": core_seed,
            "core": core,
            "fresh": fresh,
            "core_state_after_pretraining": state,
            "train_features": train_features,
            "selection_features": selection_features,
            "mechanistic": mechanistic,
            "mechanistic_score": (
                mechanistic["action_binding_loss_increase"]),
            "pretraining": pretraining,
            "stages": [],
            "optimizer_updates": 0,
            "examples_processed": 0,
        }

    active = list(horses)
    stage_survivors = {}
    final_models = {}
    for stage in STAGES:
        stage_records = {}
        for name in active:
            horse = horses[name]
            model = _fit(
                initial, horse["train_features"], attempted, rewards,
                prefix=stage, updates=args.fit_updates,
                batch_size=args.batch_size)
            audit = _stage_audit(
                model, horse["selection_features"], selection_data)
            floor = causal_floor(audit)
            previous = (
                horse["stages"][-1]["causal_floor"]
                if horse["stages"] else 0.0)
            point = {
                "unique_reward_bits": stage,
                "optimizer_updates": args.fit_updates,
                "examples_processed": (
                    args.fit_updates * min(args.batch_size, stage)),
                "audit": audit,
                "causal_floor": floor,
                "progress": floor - previous,
            }
            horse["stages"].append(point)
            horse["optimizer_updates"] += args.fit_updates
            horse["examples_processed"] += point["examples_processed"]
            stage_records[name] = {
                "causal_floor": floor,
                "progress": point["progress"],
                "mechanistic_score": horse["mechanistic_score"],
            }
            if stage == max(STAGES):
                final_models[name] = model
            print(json.dumps({
                "horse": name, "stage": stage,
                "causal_floor": floor,
                "normal_accuracy": audit["normal_accuracy"],
            }, sort_keys=True), flush=True)
        if stage == 32:
            active = select_survivors(
                active, stage_records, count=4, stage=stage)
        elif stage == 48:
            active = select_survivors(
                active, stage_records, count=3, stage=stage)
        stage_survivors[str(stage)] = list(active)
        print(json.dumps({
            "stage": stage, "survivors": active,
        }, sort_keys=True), flush=True)

    finalists = {}
    for name in active:
        horse = horses[name]
        core = horse["core"]
        model = final_models[name]
        blind_features = {
            key: decision_features(
                core, data, passive=False, device=device)
            for key, data in blind_data.items()
        }
        audit = _audit(model, blind_features, blind_data)
        fresh_train = decision_features(
            horse["fresh"], train, passive=False,
            device=device)[order.to(device)]
        fresh_blind = decision_features(
            horse["fresh"], blind_data["normal"],
            passive=False, device=device)
        action_control = _fit(
            initial, horse["train_features"],
            attempted[control_permutation], rewards,
            prefix=max(STAGES), updates=args.fit_updates,
            batch_size=args.batch_size)
        reward_control = _fit(
            initial, horse["train_features"], attempted,
            rewards[control_permutation],
            prefix=max(STAGES), updates=args.fit_updates,
            batch_size=args.batch_size)
        fresh_control = _fit(
            initial, fresh_train, attempted, rewards,
            prefix=max(STAGES), updates=args.fit_updates,
            batch_size=args.batch_size)
        controls = {
            "action_shuffled": _clean(evaluate(
                action_control, blind_features["normal"],
                blind_data["normal"]["correct_actions"])),
            "reward_shuffled": _clean(evaluate(
                reward_control, blind_features["normal"],
                blind_data["normal"]["correct_actions"])),
            "fully_fresh_core": _clean(evaluate(
                fresh_control, fresh_blind,
                blind_data["normal"]["correct_actions"])),
        }
        predictive_after = predictive_metrics(
            core, selection_data["normal"],
            passive=False, device=device)
        retention = {
            "core_parameters_bit_identical": state_is_bit_identical(
                horse["core_state_after_pretraining"],
                core.state_dict()),
            "predictive_metrics_before": horse["mechanistic"],
            "predictive_metrics_after": predictive_after,
            "passes": (
                state_is_bit_identical(
                    horse["core_state_after_pretraining"],
                    core.state_dict()) and
                predictive_after == horse["mechanistic"]),
        }
        eligible = _passes(audit, controls) and retention["passes"]
        finalists[name] = {
            "core_seed": horse["core_seed"],
            "blind_audit": audit,
            "blind_causal_floor": causal_floor(audit),
            "controls": controls,
            "retention": retention,
            "eligible": eligible,
        }

    eligible = [
        name for name, value in finalists.items()
        if value["eligible"]]
    winner = (
        max(eligible, key=lambda name:
            finalists[name]["blind_causal_floor"])
        if eligible else None)
    serializable_horses = {
        name: {
            key: value for key, value in horse.items()
            if key not in (
                "core", "fresh", "core_state_after_pretraining",
                "train_features", "selection_features")
        }
        for name, horse in horses.items()
    }
    report = {
        "schema": "core-initialization-horse-race-v1",
        "semantic_labels_used_for_training": False,
        "unattempted_action_labels_used_for_training": False,
        "same_sensory_and_verifier_experience_for_every_horse": True,
        "unique_environment_reward_bits_generated": max(STAGES),
        "search_reward_bit_presentations": sum(
            point["unique_reward_bits"]
            for horse in horses.values()
            for point in horse["stages"]),
        "retention_is_an_eligibility_gate": True,
        "late_ignition_slot_reserved": True,
        "configuration": vars(args) | {"report": str(args.report)},
        "core_seeds": list(CORE_SEEDS),
        "stage_survivors": stage_survivors,
        "horses": serializable_horses,
        "finalists": finalists,
        "eligible_finalists": eligible,
        "winner_pending_replication": winner,
        "checkpoint_promoted": False,
        "total_seconds": time.perf_counter() - started,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "eligible_finalists": eligible,
        "winner_pending_replication": winner,
        "total_seconds": report["total_seconds"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()

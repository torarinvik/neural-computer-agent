"""Behavioral temporal retention after sequential predictive-core updates."""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch

from .train import seed_everything
from .train_action_conditioned_success import frozen_final_states
from .train_actuator_transfer import (
    SuccessSystem,
    _evaluate,
    _fit_selected_success,
    uniform_logged_protocol_buffer,
)
from .train_cross_primitive_transfer import (
    SPATIAL_TRAIN_START,
    spatial_policy_sequences,
)
from .train_fixed_reward_replay_sweep import select_policy_input
from .train_third_primitive_compounding import (
    EXTRA_TEMPORAL_START,
    _trained_core,
)
from .train_zero_label_predictive_state import (
    POLICY_TEST_START,
    POLICY_TRAIN_START,
    PRETRAIN_START,
    TEST_PALETTES,
    TRAIN_PALETTES,
    PredictiveStateAgent,
    policy_sequences,
    predictive_sequences,
)


def retention_drop(reference: float, candidate: float) -> float:
    return float(reference - candidate)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--intention-width", type=int, default=8)
    parser.add_argument("--pretrain-lifetimes", type=int, default=252)
    parser.add_argument("--pretrain-steps", type=int, default=40)
    parser.add_argument("--policy-lifetimes", type=int, default=510)
    parser.add_argument("--test-lifetimes", type=int, default=384)
    parser.add_argument("--fit-updates", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--pretrain-learning-rate", type=float, default=3e-4)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    started = time.perf_counter()

    temporal_pretrain = predictive_sequences(
        PRETRAIN_START, args.pretrain_lifetimes)
    spatial_pretrain, _ = spatial_policy_sequences(
        SPATIAL_TRAIN_START, args.pretrain_lifetimes,
        heldout=False, palettes=TRAIN_PALETTES)
    extra_temporal = predictive_sequences(
        EXTRA_TEMPORAL_START, args.pretrain_lifetimes)
    train_frames, train_rules = policy_sequences(
        POLICY_TRAIN_START, args.policy_lifetimes,
        heldout=False, palettes=TRAIN_PALETTES)
    test_frames, test_rules = policy_sequences(
        POLICY_TEST_START, args.test_lifetimes,
        heldout=True, palettes=TEST_PALETTES)
    reversed_frames, reversed_rules = policy_sequences(
        POLICY_TEST_START, args.test_lifetimes,
        heldout=True, palettes=TEST_PALETTES, reverse_support_only=True)
    train_frames = select_policy_input(train_frames, "support-only")
    test_frames = select_policy_input(test_frames, "support-only")
    reversed_frames = select_policy_input(reversed_frames, "support-only")

    seed_everything(args.seed)
    initial = PredictiveStateAgent(args.hidden).to(device)
    initial_state = copy.deepcopy(initial.state_dict())
    temporal_core, temporal_accounting = _trained_core(
        initial_state, temporal_pretrain, args, device,
        shuffled=False, seed=args.seed)
    temporal_state = copy.deepcopy(temporal_core.state_dict())
    temporal_spatial, spatial_accounting = _trained_core(
        temporal_state, spatial_pretrain, args, device,
        shuffled=False, seed=args.seed + 1)
    spatial_shuffled, shuffled_accounting = _trained_core(
        temporal_state, spatial_pretrain, args, device,
        shuffled=True, seed=args.seed + 1)
    temporal_extra, extra_accounting = _trained_core(
        temporal_state, extra_temporal, args, device,
        shuffled=False, seed=args.seed + 1)
    fully_fresh = PredictiveStateAgent(args.hidden).to(device)
    fully_fresh.load_state_dict(initial_state)
    cores = {
        "temporal_only": temporal_core,
        "temporal_spatial": temporal_spatial,
        "spatial_future_shuffled": spatial_shuffled,
        "extra_temporal": temporal_extra,
        "fully_fresh": fully_fresh,
    }

    train_states = {
        name: frozen_final_states(
            core, train_frames, args.batch_size, device)
        for name, core in cores.items()}
    test_states = {
        name: frozen_final_states(
            core, test_frames, args.batch_size, device)
        for name, core in cores.items()}
    reversed_states = {
        name: frozen_final_states(
            core, reversed_frames, args.batch_size, device)
        for name, core in cores.items()}
    protocol = torch.tensor([0, 1])
    logged = {
        name: uniform_logged_protocol_buffer(
            states, train_rules, protocol, actions=2,
            seed=args.seed + 300)
        for name, states in train_states.items()}
    reference = tuple(
        item.cpu() for item in logged["temporal_only"][1:4])
    for name, output in logged.items():
        assert all(torch.equal(left, right) for left, right in zip(
            reference, (item.cpu() for item in output[1:4]))), name

    seed_everything(args.seed + 400)
    template = SuccessSystem(
        args.hidden, args.intention_width, actions=2).to(device)
    initial_head = copy.deepcopy(template.state_dict())
    arms = {}
    for name in cores:
        states, _, actions, rewards, _ = logged[name]
        seed_everything(args.seed + 500)
        head = SuccessSystem(
            args.hidden, args.intention_width, actions=2).to(device)
        head.load_state_dict(initial_head)
        accounting = _fit_selected_success(
            head, states, actions, rewards,
            updates=args.fit_updates, batch_size=args.batch_size,
            learning_rate=args.learning_rate, seed=args.seed + 600)
        normal = _evaluate(
            head, test_states[name], test_rules,
            protocol, args.batch_size)
        reversed_audit = _evaluate(
            head, reversed_states[name], reversed_rules,
            protocol, args.batch_size)
        normal_predictions = head(test_states[name]).argmax(-1).cpu()
        reversed_predictions = head(
            reversed_states[name]).argmax(-1).cpu()
        arms[name] = accounting | normal | {
            "reversed_relabeled_accuracy":
                reversed_audit["verified_accuracy"],
            "reversal_prediction_flip_rate": float(
                (normal_predictions != reversed_predictions).float().mean()),
        }
        print(json.dumps({"arm": name, **arms[name]}, sort_keys=True),
              flush=True)

    reference_arm = arms["temporal_only"]
    candidate_arm = arms["temporal_spatial"]
    drops = {
        "normal_accuracy": retention_drop(
            float(reference_arm["verified_accuracy"]),
            float(candidate_arm["verified_accuracy"])),
        "reversed_accuracy": retention_drop(
            float(reference_arm["reversed_relabeled_accuracy"]),
            float(candidate_arm["reversed_relabeled_accuracy"])),
    }
    gate = {
        "provisional_behavioral_forgetting": (
            drops["normal_accuracy"] >= 0.03 and
            drops["reversed_accuracy"] >= 0.03),
        "advance_to_rehearsal_experiment": False,
    }
    gate["advance_to_rehearsal_experiment"] = bool(
        gate["provisional_behavioral_forgetting"])
    report = {
        "schema": "predictive-core-temporal-retention-v1",
        "semantic_labels_used_for_training": False,
        "unattempted_action_labels_used_for_training": False,
        "configuration": vars(args) | {"report": str(args.report)},
        "pretraining": {
            "temporal": temporal_accounting,
            "spatial": spatial_accounting,
            "spatial_shuffled": shuffled_accounting,
            "extra_temporal": extra_accounting,
        },
        "arms": arms,
        "temporal_spatial_retention_drops": drops,
        "gate": gate,
        "total_seconds": time.perf_counter() - started,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "drops": drops, "gate": gate,
        "total_seconds": report["total_seconds"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()

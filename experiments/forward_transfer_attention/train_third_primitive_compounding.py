"""Third-primitive zero-label predictive-curriculum experiment."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .environment import _frame
from .probe_palette_sample_efficiency import _balanced_specs
from .train import seed_everything
from .train_action_conditioned_success import frozen_final_states
from .train_actuator_transfer import (
    SuccessSystem,
    _curve_aulc,
    _evaluate,
    _fit_selected_success,
    _thresholds,
    correct_protocol_actions,
    opposite_rule_permutation,
    protocol_for_seed,
    uniform_logged_protocol_buffer,
)
from .train_cross_primitive_transfer import (
    SPATIAL_TRAIN_START,
    spatial_policy_sequences,
)
from .train_zero_label_predictive_state import (
    PRETRAIN_START,
    TEST_PALETTES,
    TRAIN_PALETTES,
    PredictiveStateAgent,
    _frames,
    predictive_sequences,
    pretrain,
)


THIRD_TRAIN_START = 77_000_000
THIRD_TEST_START = 79_000_000
EXTRA_TEMPORAL_START = 75_000_000


def _balanced_binary_assignments(
        specs: list[tuple[int, tuple[int, ...]]], *,
        heldout: bool, purpose: str) -> list[int]:
    if len(specs) % 2:
        raise ValueError("binary task requires an even lifetime count")
    ranked = sorted(
        range(len(specs)),
        key=lambda index: hashlib.blake2b(
            f"third-primitive-v1:{specs[index][0]}:{int(heldout)}:"
            f"{purpose}".encode(), digest_size=16).digest())
    values = [0] * len(specs)
    for index in ranked[len(specs) // 2:]:
        values[index] = 1
    return values


def same_different_sequences(
        start: int, count: int, *, heldout: bool, palettes,
        counterfactual: bool = False,
        omit_first: bool = False, omit_second: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor]:
    if omit_first and omit_second:
        raise ValueError("cannot remove both identity frames")
    specs = list(_balanced_specs(
        start, count, palettes, heldout=heldout))
    rules = _balanced_binary_assignments(
        specs, heldout=heldout, purpose="same-different-rule")
    first_choices = _balanced_binary_assignments(
        specs, heldout=heldout, purpose="first-identity")
    sequences, private_rules = [], []
    for index, (seed, palette) in enumerate(specs):
        rule = rules[index]  # 0=same, 1=different
        first_index = first_choices[index]
        second_index = first_index if rule == 0 else 1 - first_index
        rendered_rule = rule
        if counterfactual:
            second_index = 1 - second_index
            rendered_rule = 1 - rule
        base = seed * 10_000 + 9_500
        first = _frame(
            base, colors=(palette[first_index],), cue_code=None,
            answer=None, shapes=(0,))
        second = _frame(
            base + 1, colors=(palette[second_index],), cue_code=None,
            answer=None, shapes=(0,))
        frames = []
        if not omit_first:
            frames.append(first)
        if not omit_second:
            frames.append(second)
        sequences.append([np.stack(frames)])
        private_rules.append(rendered_rule)
    return _frames(sequences), torch.tensor(
        private_rules, dtype=torch.long)


def _trained_core(initial_state, frames, args, device, *,
                  shuffled: bool, seed: int):
    agent = PredictiveStateAgent(args.hidden).to(device)
    agent.load_state_dict(initial_state)
    _, accounting = pretrain(
        agent, frames, steps=args.pretrain_steps,
        batch_size=args.batch_size,
        learning_rate=args.pretrain_learning_rate,
        shuffled=shuffled, objective="standardized",
        variance_weight=2.0, correlation_weight=0.5,
        target_kind="delta", seed=seed, device=device)
    return agent, accounting


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--intention-width", type=int, default=8)
    parser.add_argument("--commands", type=int, default=4)
    parser.add_argument("--pretrain-lifetimes", type=int, default=252)
    parser.add_argument("--pretrain-steps", type=int, default=40)
    parser.add_argument("--task-lifetimes", type=int, default=510)
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

    temporal_frames = predictive_sequences(
        PRETRAIN_START, args.pretrain_lifetimes)
    extra_temporal_frames = predictive_sequences(
        EXTRA_TEMPORAL_START, args.pretrain_lifetimes)
    spatial_frames, _ = spatial_policy_sequences(
        SPATIAL_TRAIN_START, args.pretrain_lifetimes,
        heldout=False, palettes=TRAIN_PALETTES)
    task_frames, task_rules = same_different_sequences(
        THIRD_TRAIN_START, args.task_lifetimes,
        heldout=False, palettes=TRAIN_PALETTES)
    test_frames, test_rules = same_different_sequences(
        THIRD_TEST_START, args.test_lifetimes,
        heldout=True, palettes=TEST_PALETTES)
    counterfactual_frames, counterfactual_rules = same_different_sequences(
        THIRD_TEST_START, args.test_lifetimes,
        heldout=True, palettes=TEST_PALETTES, counterfactual=True)
    missing_first_frames, missing_first_rules = same_different_sequences(
        THIRD_TEST_START, args.test_lifetimes,
        heldout=True, palettes=TEST_PALETTES, omit_first=True)
    missing_second_frames, missing_second_rules = same_different_sequences(
        THIRD_TEST_START, args.test_lifetimes,
        heldout=True, palettes=TEST_PALETTES, omit_second=True)

    seed_everything(args.seed)
    initial = PredictiveStateAgent(args.hidden).to(device)
    initial_state = copy.deepcopy(initial.state_dict())
    temporal_core, temporal_accounting = _trained_core(
        initial_state, temporal_frames, args, device,
        shuffled=False, seed=args.seed)
    temporal_state = copy.deepcopy(temporal_core.state_dict())
    temporal_spatial, spatial_accounting = _trained_core(
        temporal_state, spatial_frames, args, device,
        shuffled=False, seed=args.seed + 1)
    temporal_spatial_shuffled, shuffled_spatial_accounting = _trained_core(
        temporal_state, spatial_frames, args, device,
        shuffled=True, seed=args.seed + 1)
    extra_temporal, extra_temporal_accounting = _trained_core(
        temporal_state, extra_temporal_frames, args, device,
        shuffled=False, seed=args.seed + 1)
    fully_fresh = PredictiveStateAgent(args.hidden).to(device)
    fully_fresh.load_state_dict(initial_state)

    cores = {
        "temporal_only": temporal_core,
        "temporal_spatial": temporal_spatial,
        "spatial_future_shuffled": temporal_spatial_shuffled,
        "extra_temporal": extra_temporal,
        "fully_fresh": fully_fresh,
    }
    train_states = {
        name: frozen_final_states(
            core, task_frames, args.batch_size, device)
        for name, core in cores.items()}
    test_states = {
        name: frozen_final_states(
            core, test_frames, args.batch_size, device)
        for name, core in cores.items()}
    candidate_counterfactual = frozen_final_states(
        temporal_spatial, counterfactual_frames, args.batch_size, device)
    candidate_missing_first = frozen_final_states(
        temporal_spatial, missing_first_frames, args.batch_size, device)
    candidate_missing_second = frozen_final_states(
        temporal_spatial, missing_second_frames, args.batch_size, device)

    protocol = protocol_for_seed(args.seed + 20_000, args.commands)
    logged = {}
    reference = None
    for name, states in train_states.items():
        output = uniform_logged_protocol_buffer(
            states, task_rules, protocol, actions=args.commands,
            seed=args.seed + 300)
        logged[name] = output
        signature = tuple(item.cpu() for item in output[1:4])
        if reference is None:
            reference = signature
        else:
            assert all(torch.equal(left, right)
                       for left, right in zip(reference, signature))
    ordered_rules, attempted, rewards, propensities = logged[
        "temporal_spatial"][1:]
    action_permutation = torch.randperm(
        attempted.shape[0],
        generator=torch.Generator().manual_seed(args.seed + 401))
    reward_permutation = torch.randperm(
        rewards.shape[0],
        generator=torch.Generator().manual_seed(args.seed + 402))
    seed_everything(args.seed + 400)
    template = SuccessSystem(
        args.hidden, args.intention_width, args.commands).to(device)
    initial_head = copy.deepcopy(template.state_dict())
    prefixes = [
        value for value in (32, 128, 256, 384, 510)
        if value <= args.task_lifetimes]
    if not prefixes or prefixes[-1] != args.task_lifetimes:
        prefixes.append(args.task_lifetimes)

    arm_sources = {
        **{name: name for name in cores},
        "action_shuffled": "temporal_spatial",
        "reward_shuffled": "temporal_spatial",
    }
    arms, final_models = {}, {}
    for arm, source in arm_sources.items():
        curve = []
        states, _, base_actions, base_rewards, _ = logged[source]
        for prefix in prefixes:
            seed_everything(args.seed + 500 + prefix)
            model = SuccessSystem(
                args.hidden, args.intention_width, args.commands).to(device)
            model.load_state_dict(initial_head)
            arm_actions = base_actions[:prefix]
            arm_rewards = base_rewards[:prefix]
            if arm == "action_shuffled":
                arm_actions = attempted[
                    action_permutation.to(device)][:prefix]
            elif arm == "reward_shuffled":
                arm_rewards = rewards[
                    reward_permutation.to(device)][:prefix]
            accounting = _fit_selected_success(
                model, states[:prefix], arm_actions, arm_rewards,
                updates=args.fit_updates, batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                seed=args.seed + 600 + prefix)
            curve.append(accounting | _evaluate(
                model, test_states[source], test_rules,
                protocol, args.batch_size))
            if prefix == prefixes[-1]:
                final_models[arm] = model
        arms[arm] = {
            "curve": curve,
            "reward_aulc_above_majority": _curve_aulc(curve, 0.5),
            "unique_reward_bits_to_threshold": _thresholds(curve),
            "heldout_accuracy_final": curve[-1]["verified_accuracy"],
        }
        print(json.dumps({"arm": arm, **arms[arm]}, sort_keys=True),
              flush=True)

    candidate = final_models["temporal_spatial"]
    normal_predictions = candidate(
        test_states["temporal_spatial"]).argmax(-1).cpu()
    counterfactual_predictions = candidate(
        candidate_counterfactual).argmax(-1).cpu()
    counterfactual_targets = correct_protocol_actions(
        counterfactual_rules, protocol)
    stale = opposite_rule_permutation(test_rules)
    audit = {
        "counterfactual_accuracy": float(
            (counterfactual_predictions ==
             counterfactual_targets).float().mean()),
        "counterfactual_flip_rate": float(
            (counterfactual_predictions != normal_predictions).float().mean()),
        "missing_first_accuracy": _evaluate(
            candidate, candidate_missing_first, missing_first_rules,
            protocol, args.batch_size)["verified_accuracy"],
        "missing_second_accuracy": _evaluate(
            candidate, candidate_missing_second, missing_second_rules,
            protocol, args.batch_size)["verified_accuracy"],
        "opposite_rule_stale_accuracy": _evaluate(
            candidate,
            test_states["temporal_spatial"][stale.to(device)],
            test_rules, protocol, args.batch_size)["verified_accuracy"],
        "swapped_protocol_accuracy": _evaluate(
            candidate, test_states["temporal_spatial"], test_rules,
            protocol.flip(0), args.batch_size)["verified_accuracy"],
    }
    candidate_arm = arms["temporal_spatial"]
    controls = [arms["spatial_future_shuffled"], arms["extra_temporal"]]
    best_control_aulc = max(float(
        arm["reward_aulc_above_majority"]) for arm in controls)
    candidate_thresholds = candidate_arm[
        "unique_reward_bits_to_threshold"]
    faster_all = all(any(
        candidate_thresholds[key] is not None and (
            control["unique_reward_bits_to_threshold"][key] is None or
            int(candidate_thresholds[key]) <
            int(control["unique_reward_bits_to_threshold"][key]))
        for key in candidate_thresholds) for control in controls)
    gate = {
        "candidate_aulc_advantage": float(
            candidate_arm["reward_aulc_above_majority"]) -
            best_control_aulc,
        "fewer_reward_bits_than_both_controls": faster_all,
        "causal_audits_pass": (
            audit["counterfactual_accuracy"] >= 0.60 and
            audit["counterfactual_flip_rate"] >= 0.50 and
            audit["missing_first_accuracy"] <= 0.55 and
            audit["missing_second_accuracy"] <= 0.55 and
            audit["opposite_rule_stale_accuracy"] <= 0.40),
    }
    gate["advance_to_second_seed"] = bool(
        gate["candidate_aulc_advantage"] >= 0.03 and
        gate["fewer_reward_bits_than_both_controls"] and
        float(candidate_arm["heldout_accuracy_final"]) >= 0.60 and
        gate["causal_audits_pass"])
    report = {
        "schema": "third-primitive-predictive-curriculum-v1",
        "claim_boundary": (
            "predictive-curriculum transfer, not yet verified compounding"),
        "semantic_labels_used_for_training": False,
        "unattempted_action_labels_used_for_training": False,
        "configuration": vars(args) | {"report": str(args.report)},
        "protocol": protocol.tolist(),
        "pretraining": {
            "temporal": temporal_accounting,
            "spatial": spatial_accounting,
            "spatial_shuffled": shuffled_spatial_accounting,
            "extra_temporal": extra_temporal_accounting,
        },
        "logging": {
            "command_counts": torch.bincount(
                attempted.cpu(), minlength=args.commands).tolist(),
            "mean_propensity": float(propensities.mean()),
            "reward_rate": float(rewards.mean()),
        },
        "arms": arms,
        "causal_audit": audit,
        "gate": gate,
        "total_seconds": time.perf_counter() - started,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "gate": gate, "causal_audit": audit,
        "total_seconds": report["total_seconds"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()

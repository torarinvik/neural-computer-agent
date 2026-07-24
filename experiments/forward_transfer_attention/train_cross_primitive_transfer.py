"""Zero-label temporal-to-spatial learning-transfer microexperiment."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .environment import _frame
from .probe_palette_sample_efficiency import _balanced_specs
from .train import seed_everything
from .train_action_conditioned_success import (
    evaluate_action_head,
    frozen_final_states,
)
from .train_actuator_transfer import (
    FrozenIntentionAdapter,
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
from .train_fixed_reward_replay_sweep import select_policy_input
from .train_zero_label_predictive_state import (
    POLICY_TEST_START,
    POLICY_TRAIN_START,
    PRETRAIN_START,
    TEST_PALETTES,
    TRAIN_PALETTES,
    PredictiveStateAgent,
    _frames,
    policy_sequences,
    predictive_sequences,
    pretrain,
)


SPATIAL_TRAIN_START = 71_000_000
SPATIAL_TEST_START = 73_000_000


def spatial_policy_sequences(
        start: int, count: int, *, heldout: bool, palettes,
        mirror: bool = False, omit_feedback: bool = False
        ) -> tuple[torch.Tensor, torch.Tensor]:
    """Render simultaneous position plus selected-identity feedback."""
    specs = list(_balanced_specs(
        start, count, palettes, heldout=heldout))
    if len(specs) % 2:
        raise ValueError("spatial private rules require an even lifetime count")

    def score(seed: int, purpose: str) -> bytes:
        return hashlib.blake2b(
            f"spatial-transfer-v1:{seed}:{int(heldout)}:{purpose}".encode(),
            digest_size=16).digest()

    # Exact balance prevents majority leakage. Cryptographic ranking avoids a
    # low-complexity relationship between rule, seed, palette, or render order.
    by_rule_score = sorted(
        range(len(specs)), key=lambda index: score(
            specs[index][0], "rule-rank"))
    rules_by_index = [0] * len(specs)
    for index in by_rule_score[len(specs) // 2:]:
        rules_by_index[index] = 1
    orientations_by_index = [0] * len(specs)
    for rule in (0, 1):
        group = [index for index in range(len(specs))
                 if rules_by_index[index] == rule]
        group.sort(key=lambda index: score(
            specs[index][0], "orientation-rank"))
        for index in group[len(group) // 2:]:
            orientations_by_index[index] = 1

    sequences = []
    private_rules = []
    for index, (seed, palette) in enumerate(specs):
        rule = rules_by_index[index]
        orientation = orientations_by_index[index]
        pair = tuple(palette if orientation == 0 else palette[::-1])
        selected_color = pair[rule]
        rendered_pair = pair[::-1] if mirror else pair
        rendered_rule = 1 - rule if mirror else rule
        base = seed * 10_000 + 9_000
        frames = [_frame(
            base, colors=rendered_pair, cue_code=None, answer=None,
            shapes=(0, 0))]
        if not omit_feedback:
            frames.append(_frame(
                base + 1, colors=(selected_color,), cue_code=None,
                answer=None, shapes=(0,)))
        sequences.append([np.stack(frames)])
        private_rules.append(rendered_rule)
    return _frames(sequences), torch.tensor(
        private_rules, dtype=torch.long)


def _make_phase_b_model(
        kind: str, acquired: SuccessSystem,
        fresh_initial: dict[str, torch.Tensor],
        adapter_initial: dict[str, torch.Tensor], *,
        hidden: int, intention_width: int, commands: int,
        device: torch.device) -> nn.Module:
    if kind == "experienced_frozen":
        model = FrozenIntentionAdapter(
            copy.deepcopy(acquired.intention),
            intention_width, commands).to(device)
        model.adapter.load_state_dict(adapter_initial)
        return model
    model = SuccessSystem(hidden, intention_width, commands).to(device)
    model.load_state_dict(fresh_initial)
    if kind != "fresh":
        model.intention.load_state_dict(acquired.intention.state_dict())
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--intention-width", type=int, default=8)
    parser.add_argument("--commands", type=int, default=4)
    parser.add_argument("--pretrain-lifetimes", type=int, default=252)
    parser.add_argument("--pretrain-steps", type=int, default=40)
    parser.add_argument("--phase-a-lifetimes", type=int, default=510)
    parser.add_argument("--phase-a-updates", type=int, default=200)
    parser.add_argument("--phase-b-lifetimes", type=int, default=510)
    parser.add_argument("--phase-b-updates", type=int, default=200)
    parser.add_argument("--test-lifetimes", type=int, default=384)
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

    pretrain_frames = predictive_sequences(
        PRETRAIN_START, args.pretrain_lifetimes)
    temporal_frames, temporal_rules = policy_sequences(
        POLICY_TRAIN_START, args.phase_a_lifetimes,
        heldout=False, palettes=TRAIN_PALETTES)
    temporal_test_frames, temporal_test_rules = policy_sequences(
        POLICY_TEST_START, args.test_lifetimes,
        heldout=True, palettes=TEST_PALETTES)
    spatial_frames, spatial_rules = spatial_policy_sequences(
        SPATIAL_TRAIN_START, args.phase_b_lifetimes,
        heldout=False, palettes=TRAIN_PALETTES)
    spatial_test_frames, spatial_test_rules = spatial_policy_sequences(
        SPATIAL_TEST_START, args.test_lifetimes,
        heldout=True, palettes=TEST_PALETTES)
    mirrored_frames, mirrored_rules = spatial_policy_sequences(
        SPATIAL_TEST_START, args.test_lifetimes,
        heldout=True, palettes=TEST_PALETTES, mirror=True)
    missing_frames, missing_rules = spatial_policy_sequences(
        SPATIAL_TEST_START, args.test_lifetimes,
        heldout=True, palettes=TEST_PALETTES, omit_feedback=True)
    temporal_frames = select_policy_input(temporal_frames, "support-only")
    temporal_test_frames = select_policy_input(
        temporal_test_frames, "support-only")

    agent = PredictiveStateAgent(args.hidden).to(device)
    initial_agent_state = copy.deepcopy(agent.state_dict())
    _, predictive_accounting = pretrain(
        agent, pretrain_frames, steps=args.pretrain_steps,
        batch_size=args.batch_size,
        learning_rate=args.pretrain_learning_rate,
        shuffled=False, objective="standardized",
        variance_weight=2.0, correlation_weight=0.5,
        target_kind="delta", seed=args.seed, device=device)
    fresh_agent = PredictiveStateAgent(args.hidden).to(device)
    fresh_agent.load_state_dict(initial_agent_state)
    shuffled_agent = PredictiveStateAgent(args.hidden).to(device)
    shuffled_agent.load_state_dict(initial_agent_state)
    _, shuffled_predictive_accounting = pretrain(
        shuffled_agent, pretrain_frames, steps=args.pretrain_steps,
        batch_size=args.batch_size,
        learning_rate=args.pretrain_learning_rate,
        shuffled=True, objective="standardized",
        variance_weight=2.0, correlation_weight=0.5,
        target_kind="delta", seed=args.seed, device=device)
    temporal_states = frozen_final_states(
        agent, temporal_frames, args.batch_size, device)
    temporal_test_states = frozen_final_states(
        agent, temporal_test_frames, args.batch_size, device)
    spatial_states = frozen_final_states(
        agent, spatial_frames, args.batch_size, device)
    spatial_test_states = frozen_final_states(
        agent, spatial_test_frames, args.batch_size, device)
    mirrored_states = frozen_final_states(
        agent, mirrored_frames, args.batch_size, device)
    missing_states = frozen_final_states(
        agent, missing_frames, args.batch_size, device)
    fully_fresh_spatial_states = frozen_final_states(
        fresh_agent, spatial_frames, args.batch_size, device)
    fully_fresh_test_states = frozen_final_states(
        fresh_agent, spatial_test_frames, args.batch_size, device)
    shuffled_core_spatial_states = frozen_final_states(
        shuffled_agent, spatial_frames, args.batch_size, device)
    shuffled_core_test_states = frozen_final_states(
        shuffled_agent, spatial_test_frames, args.batch_size, device)

    # Phase A acquires the temporal intention from partial action feedback.
    phase_a_states, _, phase_a_actions, phase_a_rewards, _ = (
        uniform_logged_protocol_buffer(
            temporal_states, temporal_rules, torch.tensor([0, 1]),
            actions=2, seed=args.seed + 100))
    seed_everything(args.seed + 200)
    acquired = SuccessSystem(
        args.hidden, args.intention_width, actions=2).to(device)
    phase_a_accounting = _fit_selected_success(
        acquired, phase_a_states, phase_a_actions, phase_a_rewards,
        updates=args.phase_a_updates, batch_size=args.batch_size,
        learning_rate=args.learning_rate, seed=args.seed + 201)
    phase_a_audit = evaluate_action_head(
        acquired, temporal_test_states, temporal_test_rules, args.batch_size)

    # Phase B is a distinct simultaneous spatial relation and new protocol.
    protocol = protocol_for_seed(args.seed + 10_000, args.commands)
    ordered_states, ordered_rules, attempted, rewards, propensities = (
        uniform_logged_protocol_buffer(
            spatial_states, spatial_rules, protocol,
            actions=args.commands, seed=args.seed + 300))
    fully_fresh_ordered_states, fully_fresh_ordered_rules, \
        fully_fresh_attempted, fully_fresh_rewards, _ = (
            uniform_logged_protocol_buffer(
                fully_fresh_spatial_states, spatial_rules, protocol,
                actions=args.commands, seed=args.seed + 300))
    assert torch.equal(ordered_rules, fully_fresh_ordered_rules)
    assert torch.equal(attempted.cpu(), fully_fresh_attempted.cpu())
    assert torch.equal(rewards.cpu(), fully_fresh_rewards.cpu())
    shuffled_core_ordered_states, shuffled_core_ordered_rules, \
        shuffled_core_attempted, shuffled_core_rewards, _ = (
            uniform_logged_protocol_buffer(
                shuffled_core_spatial_states, spatial_rules, protocol,
                actions=args.commands, seed=args.seed + 300))
    assert torch.equal(ordered_rules, shuffled_core_ordered_rules)
    assert torch.equal(attempted.cpu(), shuffled_core_attempted.cpu())
    assert torch.equal(rewards.cpu(), shuffled_core_rewards.cpu())
    seed_everything(args.seed + 400)
    fresh_template = SuccessSystem(
        args.hidden, args.intention_width, args.commands).to(device)
    fresh_initial = copy.deepcopy(fresh_template.state_dict())
    adapter_initial = copy.deepcopy(fresh_template.adapter.state_dict())
    action_permutation = torch.randperm(
        attempted.shape[0],
        generator=torch.Generator().manual_seed(args.seed + 401))
    reward_permutation = torch.randperm(
        rewards.shape[0],
        generator=torch.Generator().manual_seed(args.seed + 402))
    prefixes = [
        value for value in (32, 128, 256, 384, 510)
        if value <= args.phase_b_lifetimes]
    if not prefixes or prefixes[-1] != args.phase_b_lifetimes:
        prefixes.append(args.phase_b_lifetimes)

    arms: dict[str, dict[str, object]] = {}
    final_models: dict[str, nn.Module] = {}
    for arm in (
            "experienced_tunable", "experienced_frozen", "fresh",
            "fully_fresh", "shuffled_predictive_core",
            "action_shuffled", "reward_shuffled"):
        curve = []
        for prefix in prefixes:
            seed_everything(args.seed + 500 + prefix)
            model = _make_phase_b_model(
                ("fresh" if arm in (
                    "fresh", "fully_fresh", "shuffled_predictive_core") else
                 "experienced_frozen" if arm == "experienced_frozen"
                 else "experienced_tunable"),
                acquired, fresh_initial, adapter_initial,
                hidden=args.hidden, intention_width=args.intention_width,
                commands=args.commands, device=device)
            arm_states = (
                fully_fresh_ordered_states[:prefix]
                if arm == "fully_fresh" else ordered_states[:prefix])
            arm_test_states = (
                fully_fresh_test_states
                if arm == "fully_fresh" else spatial_test_states)
            if arm == "shuffled_predictive_core":
                arm_states = shuffled_core_ordered_states[:prefix]
                arm_test_states = shuffled_core_test_states
            arm_actions = attempted[:prefix]
            arm_rewards = rewards[:prefix]
            if arm == "action_shuffled":
                arm_actions = attempted[
                    action_permutation.to(device)][:prefix]
            elif arm == "reward_shuffled":
                arm_rewards = rewards[
                    reward_permutation.to(device)][:prefix]
            accounting = _fit_selected_success(
                model, arm_states, arm_actions, arm_rewards,
                updates=args.phase_b_updates, batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                seed=args.seed + 600 + prefix)
            curve.append(accounting | _evaluate(
                model, arm_test_states, spatial_test_rules,
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

    candidate = final_models["experienced_tunable"]
    normal_predictions = candidate(spatial_test_states).argmax(-1).cpu()
    mirrored_predictions = candidate(mirrored_states).argmax(-1).cpu()
    mirrored_targets = correct_protocol_actions(mirrored_rules, protocol)
    stale_permutation = opposite_rule_permutation(spatial_test_rules)
    causal_audit = {
        "mirrored_relabeled_accuracy": float(
            (mirrored_predictions == mirrored_targets).float().mean()),
        "mirrored_prediction_flip_rate": float(
            (mirrored_predictions != normal_predictions).float().mean()),
        "missing_feedback_accuracy": _evaluate(
            candidate, missing_states, missing_rules,
            protocol, args.batch_size)["verified_accuracy"],
        "opposite_rule_stale_accuracy": _evaluate(
            candidate,
            spatial_test_states[stale_permutation.to(device)],
            spatial_test_rules, protocol,
            args.batch_size)["verified_accuracy"],
        "swapped_protocol_accuracy": _evaluate(
            candidate, spatial_test_states, spatial_test_rules,
            protocol.flip(0), args.batch_size)["verified_accuracy"],
    }
    candidate_arm = arms["experienced_tunable"]
    candidate_aulc = float(
        candidate_arm["reward_aulc_above_majority"])
    best_control_aulc = max(
        float(arms[name]["reward_aulc_above_majority"])
        for name in ("experienced_frozen", "fresh", "fully_fresh",
                     "shuffled_predictive_core",
                     "action_shuffled", "reward_shuffled"))
    candidate_thresholds = candidate_arm[
        "unique_reward_bits_to_threshold"]
    fresh_thresholds = arms["fresh"][
        "unique_reward_bits_to_threshold"]
    faster = any(
        candidate_thresholds[key] is not None and (
            fresh_thresholds[key] is None or
            int(candidate_thresholds[key]) < int(fresh_thresholds[key]))
        for key in candidate_thresholds)
    gate = {
        "candidate_aulc_advantage": candidate_aulc - best_control_aulc,
        "fewer_reward_bits_than_fresh": faster,
        "mirror_causality": (
            causal_audit["mirrored_relabeled_accuracy"] >= 0.60 and
            causal_audit["mirrored_prediction_flip_rate"] >= 0.50),
        "missing_feedback_degrades": (
            causal_audit["missing_feedback_accuracy"] <= 0.55),
        "stale_state_degrades": (
            causal_audit["opposite_rule_stale_accuracy"] <= 0.40),
    }
    gate["advance_to_second_seed"] = bool(
        gate["candidate_aulc_advantage"] >= 0.03 and
        float(candidate_arm["heldout_accuracy_final"]) >= 0.60 and
        gate["fewer_reward_bits_than_fresh"] and
        gate["mirror_causality"] and
        gate["missing_feedback_degrades"] and
        gate["stale_state_degrades"])
    core_candidate = arms["fresh"]
    core_control = arms["shuffled_predictive_core"]
    core_thresholds = core_candidate[
        "unique_reward_bits_to_threshold"]
    shuffled_core_thresholds = core_control[
        "unique_reward_bits_to_threshold"]
    core_faster = any(
        core_thresholds[key] is not None and (
            shuffled_core_thresholds[key] is None or
            int(core_thresholds[key]) < int(shuffled_core_thresholds[key]))
        for key in core_thresholds)
    core_model = final_models["fresh"]
    core_normal = core_model(spatial_test_states).argmax(-1).cpu()
    core_mirrored = core_model(mirrored_states).argmax(-1).cpu()
    core_causal_audit = {
        "mirrored_relabeled_accuracy": float(
            (core_mirrored == mirrored_targets).float().mean()),
        "mirrored_prediction_flip_rate": float(
            (core_mirrored != core_normal).float().mean()),
        "missing_feedback_accuracy": _evaluate(
            core_model, missing_states, missing_rules,
            protocol, args.batch_size)["verified_accuracy"],
        "opposite_rule_stale_accuracy": _evaluate(
            core_model,
            spatial_test_states[stale_permutation.to(device)],
            spatial_test_rules, protocol,
            args.batch_size)["verified_accuracy"],
    }
    core_transfer_gate = {
        "paired_core_aulc_advantage": (
            float(core_candidate["reward_aulc_above_majority"]) -
            float(core_control["reward_aulc_above_majority"])),
        "fewer_reward_bits_than_shuffled_core": core_faster,
        "final_accuracy": float(core_candidate["heldout_accuracy_final"]),
        "causal_audits_pass": (
            core_causal_audit["mirrored_relabeled_accuracy"] >= 0.60 and
            core_causal_audit["mirrored_prediction_flip_rate"] >= 0.50 and
            core_causal_audit["missing_feedback_accuracy"] <= 0.55 and
            core_causal_audit["opposite_rule_stale_accuracy"] <= 0.40),
    }
    core_transfer_gate["advance_to_second_seed"] = bool(
        core_transfer_gate["paired_core_aulc_advantage"] >= 0.03 and
        core_transfer_gate["fewer_reward_bits_than_shuffled_core"] and
        core_transfer_gate["final_accuracy"] >= 0.60 and
        core_transfer_gate["causal_audits_pass"])
    report = {
        "schema": "zero-label-cross-primitive-transfer-v1",
        "claim_boundary": (
            "temporal-to-spatial exploratory transfer; compounding requires "
            "at least six primitives"),
        "semantic_labels_used_for_training": False,
        "unattempted_action_labels_used_for_training": False,
        "learner_visible": [
            "rendered_rgb_stream", "recurrent_state", "attempted_action",
            "uniform_logging_propensity", "scalar_observed_reward"],
        "verifier_private": [
            "temporal_rule", "spatial_rule", "correct_protocol_command",
            "object_identity", "palette", "logical_lifetime_metadata"],
        "configuration": vars(args) | {"report": str(args.report)},
        "protocol": protocol.tolist(),
        "predictive_pretraining": predictive_accounting,
        "shuffled_future_pretraining": shuffled_predictive_accounting,
        "phase_a": {
            "accounting": phase_a_accounting,
            "heldout_audit": phase_a_audit,
        },
        "phase_b_logging": {
            "command_counts": torch.bincount(
                attempted.cpu(), minlength=args.commands).tolist(),
            "mean_propensity": float(propensities.mean()),
            "reward_rate": float(rewards.mean()),
        },
        "arms": arms,
        "causal_audit": causal_audit,
        "gate": gate,
        "core_causal_audit": core_causal_audit,
        "core_transfer_gate": core_transfer_gate,
        "total_seconds": time.perf_counter() - started,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "gate": gate, "causal_audit": causal_audit,
        "core_transfer_gate": core_transfer_gate,
        "core_causal_audit": core_causal_audit,
        "total_seconds": report["total_seconds"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()

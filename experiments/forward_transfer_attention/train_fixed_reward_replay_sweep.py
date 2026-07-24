"""Fixed-reward-bit replay sweep for the temporal contextual bandit.

Uniformly explored `(state, attempted action, scalar reward)` tuples are logged
once.  Heads may replay those observations, but never receive the correct
action or a target for an unattempted action.  Unique reward bits and gradient
compute are reported separately.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch

from .train import seed_everything
from .train_action_conditioned_success import (
    NonlinearActionHead,
    evaluate_action_head,
    frozen_final_states,
    reversal_audit,
    selected_success_loss,
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
    pretrain,
)


def uniform_logged_buffer(states: torch.Tensor,
                          private_rules: torch.Tensor, *,
                          seed: int
                          ) -> tuple[torch.Tensor, ...]:
    """Create balanced exploration without consulting the private rule."""
    count = states.shape[0]
    order = torch.randperm(
        count, generator=torch.Generator().manual_seed(seed + 11))
    states = states[order.to(states.device)]
    rules = private_rules[order]
    actions = (torch.arange(count) % 2).to(states.device)
    rewards = (actions.cpu() == rules).to(states.device, states.dtype)
    propensities = torch.full(
        (count,), 0.5, device=states.device, dtype=states.dtype)
    return states, rules, actions, rewards, propensities


def select_policy_input(frames: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "full":
        return frames
    if mode == "support-only":
        # Rule-identification microtask: the query cannot supply an alternate
        # route from feedback identity to the private first/last rule.
        return frames[:, :3]
    raise ValueError(f"unknown input mode {mode!r}")


def _fit_head(
        initial_state: dict[str, torch.Tensor],
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor, *,
        hidden: int, updates: int, batch_size: int,
        learning_rate: float, mode: str, seed: int
        ) -> tuple[NonlinearActionHead, dict[str, float | int]]:
    seed_everything(seed)
    device = states.device
    head = NonlinearActionHead(hidden).to(device)
    head.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=learning_rate, weight_decay=1e-4)
    generator = torch.Generator(device=device).manual_seed(seed + 29)
    last_loss = 0.0
    last_gradient_norm = 0.0
    for _ in range(updates):
        indices = torch.randint(
            states.shape[0],
            (min(batch_size, states.shape[0]),),
            generator=generator, device=device)
        logits = head(states[indices])
        selected_actions = actions[indices]
        selected_rewards = rewards[indices]
        if mode == "success":
            loss = selected_success_loss(
                logits, selected_actions, selected_rewards)
        elif mode == "ips":
            probabilities = torch.softmax(logits, dim=-1)
            selected = probabilities.gather(
                1, selected_actions[:, None]).squeeze(1)
            # Logging propensity is exactly 0.5. This objective uses only the
            # reward observed for the attempted action.
            ips_return = selected / 0.5 * selected_rewards
            entropy = -(
                probabilities * probabilities.clamp_min(1e-8).log()
            ).sum(-1).mean()
            loss = -ips_return.mean() - 0.01 * entropy
        else:
            raise ValueError(f"unknown mode {mode!r}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        last_gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0))
        optimizer.step()
        last_loss = float(loss.detach())
    return head, {
        "optimizer_updates": updates,
        "examples_processed": updates * min(batch_size, states.shape[0]),
        "unique_reward_bits": states.shape[0],
        "last_loss": last_loss,
        "last_gradient_norm": last_gradient_norm,
    }


def _curve_aulc(curve: list[dict[str, object]]) -> float:
    return sum(
        max(0.0, float(point["verified_accuracy"]) - 0.5)
        for point in curve) / len(curve)


def _thresholds(curve: list[dict[str, object]]) -> dict[str, int | None]:
    return {
        str(threshold): next((
            int(first["unique_reward_bits"])
            for first, second in zip(curve, curve[1:])
            if (float(first["verified_accuracy"]) >= threshold
                and float(second["verified_accuracy"]) >= threshold)), None)
        for threshold in (0.60, 0.70, 0.80)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--pretrain-lifetimes", type=int, default=252)
    parser.add_argument("--pretrain-steps", type=int, default=40)
    parser.add_argument("--policy-lifetimes", type=int, default=510)
    parser.add_argument("--policy-test-lifetimes", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--fit-updates", type=int, default=200)
    parser.add_argument(
        "--input-mode", choices=("full", "support-only"), default="full")
    parser.add_argument("--pretrain-learning-rate", type=float, default=3e-4)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--variance-weight", type=float, default=2.0)
    parser.add_argument("--correlation-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    started = time.perf_counter()

    pretrain_frames = predictive_sequences(
        PRETRAIN_START, args.pretrain_lifetimes)
    train_frames, train_rules = policy_sequences(
        POLICY_TRAIN_START, args.policy_lifetimes, heldout=False,
        palettes=TRAIN_PALETTES)
    test_frames, test_rules = policy_sequences(
        POLICY_TEST_START, args.policy_test_lifetimes, heldout=True,
        palettes=TEST_PALETTES)
    reversed_frames, reversed_rules = policy_sequences(
        POLICY_TEST_START, args.policy_test_lifetimes, heldout=True,
        palettes=TEST_PALETTES, reverse_events=True)
    train_frames = select_policy_input(train_frames, args.input_mode)
    test_frames = select_policy_input(test_frames, args.input_mode)
    reversed_frames = select_policy_input(reversed_frames, args.input_mode)
    data_seconds = time.perf_counter() - started

    seed_everything(args.seed)
    fresh_agent = PredictiveStateAgent(args.hidden).to(device)
    initial_agent_state = copy.deepcopy(fresh_agent.state_dict())
    delta_agent = PredictiveStateAgent(args.hidden).to(device)
    delta_agent.load_state_dict(initial_agent_state)
    _, pretraining = pretrain(
        delta_agent, pretrain_frames, steps=args.pretrain_steps,
        batch_size=args.batch_size,
        learning_rate=args.pretrain_learning_rate,
        shuffled=False, objective="standardized",
        variance_weight=args.variance_weight,
        correlation_weight=args.correlation_weight,
        target_kind="delta", seed=args.seed, device=device)
    shuffled_agent = PredictiveStateAgent(args.hidden).to(device)
    shuffled_agent.load_state_dict(initial_agent_state)
    _, shuffled_pretraining = pretrain(
        shuffled_agent, pretrain_frames, steps=args.pretrain_steps,
        batch_size=args.batch_size,
        learning_rate=args.pretrain_learning_rate,
        shuffled=True, objective="standardized",
        variance_weight=args.variance_weight,
        correlation_weight=args.correlation_weight,
        target_kind="delta", seed=args.seed, device=device)

    delta_train = frozen_final_states(
        delta_agent, train_frames, args.batch_size, device)
    delta_test = frozen_final_states(
        delta_agent, test_frames, args.batch_size, device)
    delta_reversed = frozen_final_states(
        delta_agent, reversed_frames, args.batch_size, device)
    fresh_train = frozen_final_states(
        fresh_agent, train_frames, args.batch_size, device)
    fresh_test = frozen_final_states(
        fresh_agent, test_frames, args.batch_size, device)
    shuffled_train = frozen_final_states(
        shuffled_agent, train_frames, args.batch_size, device)
    shuffled_test = frozen_final_states(
        shuffled_agent, test_frames, args.batch_size, device)

    delta_train, ordered_rules, actions, rewards, propensities = (
        uniform_logged_buffer(
            delta_train, train_rules, seed=args.seed + 300))
    # Apply the exact same lifetime ordering and attempted actions to fresh
    # states. The permutation is reconstructed without consulting rules.
    fresh_train, fresh_ordered_rules, fresh_actions, fresh_rewards, _ = (
        uniform_logged_buffer(
            fresh_train, train_rules, seed=args.seed + 300))
    assert torch.equal(ordered_rules, fresh_ordered_rules)
    assert torch.equal(actions.cpu(), fresh_actions.cpu())
    assert torch.equal(rewards.cpu(), fresh_rewards.cpu())
    shuffled_train, shuffled_ordered_rules, shuffled_rep_actions, \
        shuffled_rep_rewards, _ = uniform_logged_buffer(
            shuffled_train, train_rules, seed=args.seed + 300)
    assert torch.equal(ordered_rules, shuffled_ordered_rules)
    assert torch.equal(actions.cpu(), shuffled_rep_actions.cpu())
    assert torch.equal(rewards.cpu(), shuffled_rep_rewards.cpu())

    permutation = torch.randperm(
        actions.shape[0],
        generator=torch.Generator().manual_seed(args.seed + 401))
    shuffled_actions = actions[permutation.to(device)]
    reward_permutation = torch.randperm(
        rewards.shape[0],
        generator=torch.Generator().manual_seed(args.seed + 402))
    shuffled_rewards = rewards[reward_permutation.to(device)]

    seed_everything(args.seed + 700)
    initial_head = NonlinearActionHead(args.hidden).to(device)
    initial_head_state = copy.deepcopy(initial_head.state_dict())
    prefixes = [
        value for value in (30, 120, 240, 360, 510)
        if value <= args.policy_lifetimes]
    if prefixes[-1] != args.policy_lifetimes:
        prefixes.append(args.policy_lifetimes)

    arm_specs = {
        "success_delta": (
            delta_train, actions, rewards, delta_test, "success"),
        "ips_delta": (
            delta_train, actions, rewards, delta_test, "ips"),
        "action_shuffled_delta": (
            delta_train, shuffled_actions, rewards, delta_test, "success"),
        "reward_shuffled_delta": (
            delta_train, actions, shuffled_rewards, delta_test, "success"),
        "success_fresh": (
            fresh_train, fresh_actions, fresh_rewards, fresh_test, "success"),
        "success_shuffled_representation": (
            shuffled_train, shuffled_rep_actions, shuffled_rep_rewards,
            shuffled_test, "success"),
    }
    arms: dict[str, dict[str, object]] = {}
    final_heads: dict[str, NonlinearActionHead] = {}
    for name, (states, arm_actions, arm_rewards, heldout, mode) in (
            arm_specs.items()):
        arm_started = time.perf_counter()
        curve = []
        for prefix in prefixes:
            head, accounting = _fit_head(
                initial_head_state, states[:prefix],
                arm_actions[:prefix], arm_rewards[:prefix],
                hidden=args.hidden, updates=args.fit_updates,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate, mode=mode,
                seed=args.seed + 800 + prefix)
            audit = evaluate_action_head(
                head, heldout, test_rules, args.batch_size)
            curve.append(accounting | audit)
            if prefix == prefixes[-1]:
                final_heads[name] = head
        arms[name] = {
            "curve": curve,
            "reward_aulc_above_chance": _curve_aulc(curve),
            "unique_reward_bits_to_threshold": _thresholds(curve),
            "heldout_accuracy_final": curve[-1]["verified_accuracy"],
            "seconds": time.perf_counter() - arm_started,
        }
        print(json.dumps({"arm": name, **arms[name]},
                         sort_keys=True), flush=True)

    compute_sweep = []
    for updates in (17, 68, args.fit_updates):
        if any(point["optimizer_updates"] == updates
               for point in compute_sweep):
            continue
        head, accounting = _fit_head(
            initial_head_state, delta_train, actions, rewards,
            hidden=args.hidden, updates=updates,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate, mode="success",
            seed=args.seed + 900)
        compute_sweep.append(accounting | evaluate_action_head(
            head, delta_test, test_rules, args.batch_size))

    candidate = arms["success_delta"]
    controls = [
        arms[name] for name in (
            "ips_delta", "action_shuffled_delta",
            "reward_shuffled_delta", "success_fresh",
            "success_shuffled_representation")]
    best_control_aulc = max(
        float(control["reward_aulc_above_chance"])
        for control in controls)
    best_control_final = max(
        float(control["heldout_accuracy_final"]) for control in controls)
    candidate_reversal = reversal_audit(
        final_heads["success_delta"], delta_test, test_rules,
        delta_reversed, reversed_rules, args.batch_size)
    gate = {
        "candidate_aulc_advantage": (
            float(candidate["reward_aulc_above_chance"]) -
            best_control_aulc),
        "candidate_beats_controls": (
            float(candidate["reward_aulc_above_chance"]) -
            best_control_aulc >= 0.02
            and float(candidate["heldout_accuracy_final"]) >= 0.60
            and float(candidate["heldout_accuracy_final"]) >
            best_control_final),
        "reversal_causality": (
            candidate_reversal["reversed_relabeled_accuracy"] >= 0.60
            and candidate_reversal["prediction_flip_rate"] >= 0.50),
        "advance_to_second_seed": False,
        "advance_to_three_minutes": False,
    }
    gate["advance_to_second_seed"] = bool(
        gate["candidate_beats_controls"] and gate["reversal_causality"])
    report = {
        "schema": "fixed-reward-replay-sweep-v3",
        "learner_visible": [
            "rendered_rgb_stream", "frozen_recurrent_latent",
            "attempted_action", "logging_propensity_0.5",
            "scalar_verifier_reward"],
        "verifier_private": [
            "correct_action", "temporal_rule", "object_identity",
            "palette_id", "event_index", "logical_lifetime_metadata"],
        "offline_audit_only": [
            "all-action_brier_targets", "counterfactual_reversal_labels"],
        "forbidden": [
            "unattempted_action_target", "inferred_other_action_label",
            "semantic_rule_target", "game_state", "task_id"],
        "semantic_labels_used_for_training": False,
        "unattempted_action_labels_used_for_training": False,
        "configuration": vars(args) | {"report": str(args.report)},
        "data_generation_seconds": data_seconds,
        "pretraining": pretraining,
        "shuffled_future_pretraining": shuffled_pretraining,
        "logging_policy": {
            "kind": "balanced uniform exploration independent of private rule",
            "action_counts": torch.bincount(
                actions.cpu(), minlength=2).tolist(),
            "mean_propensity": float(propensities.mean()),
            "reward_rate": float(rewards.mean()),
        },
        "arms": arms,
        "full_buffer_compute_sweep": compute_sweep,
        "candidate_reversal_audit": candidate_reversal,
        "gate": gate,
        "total_seconds": time.perf_counter() - started,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "gate": gate,
        "candidate_reversal_audit": candidate_reversal,
        "total_seconds": report["total_seconds"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()

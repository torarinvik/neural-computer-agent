"""Attempted-action-only success prediction versus matched REINFORCE.

The learner sees rendered RGB, its sampled action, the logging propensity, and
the resulting scalar reward.  Correct actions and semantic rule labels are
private verifier facts and never become differentiable targets.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch
from torch import nn

from .train import seed_everything
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
    representation_metrics,
)


class NonlinearActionHead(nn.Module):
    """Small matched-capacity head used by both learning algorithms."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.network(states)


class ValueHead(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.network(states).squeeze(-1)


def exploration_probabilities(logits: torch.Tensor,
                              epsilon: float) -> torch.Tensor:
    """Differentiable epsilon-mixture with a known logging propensity."""
    probabilities = torch.softmax(logits, dim=-1)
    return probabilities * (1.0 - epsilon) + epsilon / logits.shape[-1]


def selected_success_loss(logits: torch.Tensor,
                          attempted_actions: torch.Tensor,
                          observed_rewards: torch.Tensor) -> torch.Tensor:
    """BCE only for attempted actions; unobserved outcomes are not targets."""
    attempted_logits = logits.gather(
        1, attempted_actions[:, None]).squeeze(1)
    return nn.functional.binary_cross_entropy_with_logits(
        attempted_logits, observed_rewards)


def expected_calibration_error(probabilities: torch.Tensor,
                               outcomes: torch.Tensor,
                               bins: int = 10) -> float:
    probabilities = probabilities.float().flatten()
    outcomes = outcomes.float().flatten()
    error = probabilities.new_zeros(())
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        if index + 1 == bins:
            mask = (probabilities >= low) & (probabilities <= high)
        else:
            mask = (probabilities >= low) & (probabilities < high)
        if mask.any():
            error += mask.float().mean() * (
                probabilities[mask].mean() - outcomes[mask].mean()).abs()
    return float(error)


@torch.no_grad()
def frozen_final_states(agent: PredictiveStateAgent, frames: torch.Tensor,
                        batch_size: int,
                        device: torch.device) -> torch.Tensor:
    agent.eval()
    parts = []
    for offset in range(0, frames.shape[0], batch_size):
        batch = frames[offset:offset + batch_size].to(device)
        parts.append(agent.states(batch)[:, -1])
    return torch.cat(parts)


@torch.no_grad()
def evaluate_action_head(head: NonlinearActionHead,
                         states: torch.Tensor,
                         private_rules: torch.Tensor,
                         batch_size: int) -> dict[str, float]:
    """Private verifier audit. None of these labels enter optimization."""
    head.eval()
    logits = torch.cat([
        head(states[offset:offset + batch_size])
        for offset in range(0, states.shape[0], batch_size)
    ])
    probabilities = torch.sigmoid(logits)
    actions = logits.argmax(-1).cpu()
    rules = private_rules.cpu()
    outcomes = torch.stack((rules == 0, rules == 1), dim=1).to(
        probabilities.device, probabilities.dtype)
    return {
        "verified_accuracy": float((actions == rules).float().mean()),
        "brier_all_action_audit": float(
            (probabilities - outcomes).square().mean()),
        "ece_all_action_audit": expected_calibration_error(
            probabilities, outcomes),
        "mean_action_entropy": float(
            torch.distributions.Categorical(
                logits=logits).entropy().mean()),
    }


def _thresholds(history: list[dict[str, float]]) -> dict[str, int | None]:
    return {
        str(threshold): next((
            int(first["unique_lifetimes"])
            for first, second in zip(history, history[1:])
            if (first["verified_accuracy"] >= threshold and
                second["verified_accuracy"] >= threshold)), None)
        for threshold in (0.60, 0.70, 0.80)
    }


def _finish_history(history: list[dict[str, float]]) -> dict[str, object]:
    chance = 0.5
    return {
        "history": history,
        "heldout_accuracy_final": history[-1]["verified_accuracy"],
        "reward_aulc_above_chance": sum(
            max(0.0, row["verified_accuracy"] - chance)
            for row in history) / len(history),
        "unique_lifetimes_to_threshold": _thresholds(history),
    }


def train_success_replay(
        head: NonlinearActionHead,
        train_states: torch.Tensor,
        private_rules: torch.Tensor,
        test_states: torch.Tensor,
        test_private_rules: torch.Tensor, *,
        batch_size: int, learning_rate: float, epsilon: float,
        shuffle_attempted_actions: bool, seed: int
        ) -> dict[str, object]:
    """One replay update per interaction batch, using attempted outcomes only."""
    seed_everything(seed)
    device = train_states.device
    head.train()
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=learning_rate, weight_decay=1e-4)
    action_generator = torch.Generator(device=device).manual_seed(seed + 177)
    cpu_generator = torch.Generator().manual_seed(seed + 313)
    order = torch.randperm(train_states.shape[0], generator=cpu_generator)
    total_updates = math.ceil(train_states.shape[0] / batch_size)
    evaluation_updates = {
        1, max(1, total_updates // 4), max(1, total_updates // 2),
        max(1, 3 * total_updates // 4), total_updates}
    history = [{
        "unique_lifetimes": 0,
        **evaluate_action_head(
            head, test_states, test_private_rules, batch_size),
    }]
    replay_states = []
    replay_actions = []
    replay_rewards = []
    replay_propensities = []
    action_counts = torch.zeros(2, dtype=torch.long, device=device)
    last_gradient_norm = 0.0
    seen = 0
    update = 0
    while seen < train_states.shape[0]:
        end = min(seen + batch_size, train_states.shape[0])
        indices = order[seen:end].to(device)
        states = train_states[indices]
        rules = private_rules[indices.cpu()].to(device)
        with torch.no_grad():
            logits = head(states)
            behavior = exploration_probabilities(logits, epsilon)
            actions = torch.multinomial(
                behavior, 1, generator=action_generator).squeeze(-1)
            propensities = behavior.gather(
                1, actions[:, None]).squeeze(1)
            rewards = (actions == rules).to(logits.dtype)
        action_counts += torch.bincount(actions, minlength=2)
        replay_states.append(states.detach())
        replay_actions.append(actions.detach())
        replay_rewards.append(rewards.detach())
        replay_propensities.append(propensities.detach())

        buffer_states = torch.cat(replay_states)
        buffer_actions = torch.cat(replay_actions)
        buffer_rewards = torch.cat(replay_rewards)
        replay_indices = torch.randint(
            buffer_states.shape[0],
            (min(batch_size, buffer_states.shape[0]),),
            generator=action_generator, device=device)
        attempted = buffer_actions[replay_indices]
        if shuffle_attempted_actions and attempted.numel() > 1:
            attempted = attempted.roll(1)
        loss = selected_success_loss(
            head(buffer_states[replay_indices]),
            attempted,
            buffer_rewards[replay_indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        last_gradient_norm = float(
            nn.utils.clip_grad_norm_(head.parameters(), 1.0))
        optimizer.step()
        seen = end
        update += 1
        if update in evaluation_updates:
            audit = evaluate_action_head(
                head, test_states, test_private_rules, batch_size)
            history.append({
                "unique_lifetimes": seen,
                **audit,
                "loss": float(loss.detach()),
                "mean_train_reward": float(rewards.mean()),
            })
            head.train()
    total_actions = int(action_counts.sum())
    coverage = (action_counts.float() / max(1, total_actions)).cpu().tolist()
    return _finish_history(history) | {
        "optimizer_updates": update,
        "examples_processed": min(
            batch_size, train_states.shape[0]) * update,
        "logged_reward_bits": train_states.shape[0],
        "action_counts": action_counts.cpu().tolist(),
        "action_fractions": coverage,
        "minimum_action_fraction": min(coverage),
        "mean_logging_propensity": float(
            torch.cat(replay_propensities).mean()),
        "last_gradient_norm": last_gradient_norm,
        "training_mode": (
            "attempted actions permuted in replay loss"
            if shuffle_attempted_actions
            else "attempted-action-only replay"),
    }


def train_matched_reinforce(
        action_head: NonlinearActionHead,
        value_head: ValueHead,
        train_states: torch.Tensor,
        private_rules: torch.Tensor,
        test_states: torch.Tensor,
        test_private_rules: torch.Tensor, *,
        batch_size: int, learning_rate: float, epsilon: float,
        seed: int) -> dict[str, object]:
    """REINFORCE with the same head, exploration, interactions, and updates."""
    seed_everything(seed)
    device = train_states.device
    parameters = [*action_head.parameters(), *value_head.parameters()]
    optimizer = torch.optim.AdamW(
        parameters, lr=learning_rate, weight_decay=1e-4)
    action_generator = torch.Generator(device=device).manual_seed(seed + 177)
    order = torch.randperm(
        train_states.shape[0],
        generator=torch.Generator().manual_seed(seed + 313))
    total_updates = math.ceil(train_states.shape[0] / batch_size)
    evaluation_updates = {
        1, max(1, total_updates // 4), max(1, total_updates // 2),
        max(1, 3 * total_updates // 4), total_updates}
    history = [{
        "unique_lifetimes": 0,
        **evaluate_action_head(
            action_head, test_states, test_private_rules, batch_size),
    }]
    action_counts = torch.zeros(2, dtype=torch.long, device=device)
    propensities = []
    seen = 0
    update = 0
    last_gradient_norm = 0.0
    while seen < train_states.shape[0]:
        end = min(seen + batch_size, train_states.shape[0])
        indices = order[seen:end].to(device)
        states = train_states[indices]
        rules = private_rules[indices.cpu()].to(device)
        logits = action_head(states)
        behavior = exploration_probabilities(logits, epsilon)
        actions = torch.multinomial(
            behavior, 1, generator=action_generator).squeeze(-1)
        logged = behavior.gather(1, actions[:, None]).squeeze(1)
        rewards = (actions == rules).to(logits.dtype)
        action_counts += torch.bincount(actions, minlength=2)
        propensities.append(logged.detach())
        advantage = rewards - rewards.mean()
        actor = -(logged.clamp_min(1e-8).log() * advantage).mean()
        values = value_head(states)
        critic = 0.5 * (values - rewards).square().mean()
        entropy = -(behavior * behavior.clamp_min(1e-8).log()).sum(-1).mean()
        loss = actor + critic - 0.01 * entropy
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        last_gradient_norm = float(
            nn.utils.clip_grad_norm_(parameters, 1.0))
        optimizer.step()
        seen = end
        update += 1
        if update in evaluation_updates:
            history.append({
                "unique_lifetimes": seen,
                **evaluate_action_head(
                    action_head, test_states, test_private_rules, batch_size),
                "loss": float(loss.detach()),
                "mean_train_reward": float(rewards.mean()),
            })
            action_head.train()
            value_head.train()
    total_actions = int(action_counts.sum())
    coverage = (action_counts.float() / max(1, total_actions)).cpu().tolist()
    return _finish_history(history) | {
        "optimizer_updates": update,
        "examples_processed": train_states.shape[0],
        "logged_reward_bits": train_states.shape[0],
        "action_counts": action_counts.cpu().tolist(),
        "action_fractions": coverage,
        "minimum_action_fraction": min(coverage),
        "mean_logging_propensity": float(torch.cat(propensities).mean()),
        "last_gradient_norm": last_gradient_norm,
        "training_mode": "matched epsilon-mixture REINFORCE",
    }


@torch.no_grad()
def reversal_audit(head: NonlinearActionHead,
                   normal_states: torch.Tensor,
                   normal_rules: torch.Tensor,
                   reversed_states: torch.Tensor,
                   reversed_rules: torch.Tensor,
                   batch_size: int) -> dict[str, float]:
    head.eval()
    normal = torch.cat([
        head(normal_states[offset:offset + batch_size]).argmax(-1)
        for offset in range(0, normal_states.shape[0], batch_size)
    ]).cpu()
    reversed_predictions = torch.cat([
        head(reversed_states[offset:offset + batch_size]).argmax(-1)
        for offset in range(0, reversed_states.shape[0], batch_size)
    ]).cpu()
    return {
        "normal_accuracy": float(
            (normal == normal_rules).float().mean()),
        "reversed_relabeled_accuracy": float(
            (reversed_predictions == reversed_rules).float().mean()),
        "reversed_stale_accuracy": float(
            (reversed_predictions == normal_rules).float().mean()),
        "prediction_flip_rate": float(
            (normal != reversed_predictions).float().mean()),
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
    parser.add_argument("--pretrain-learning-rate", type=float, default=3e-4)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--epsilon", type=float, default=0.20)
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
    data_seconds = time.perf_counter() - started

    seed_everything(args.seed)
    fresh_agent = PredictiveStateAgent(args.hidden).to(device)
    initial_state = copy.deepcopy(fresh_agent.state_dict())
    delta_agent = PredictiveStateAgent(args.hidden).to(device)
    delta_agent.load_state_dict(initial_state)
    target, pretraining = pretrain(
        delta_agent, pretrain_frames, steps=args.pretrain_steps,
        batch_size=args.batch_size,
        learning_rate=args.pretrain_learning_rate,
        shuffled=False, objective="standardized",
        variance_weight=args.variance_weight,
        correlation_weight=args.correlation_weight,
        target_kind="delta", seed=args.seed, device=device)
    representation = representation_metrics(
        delta_agent, target, pretrain_frames[:96].to(device),
        objective="standardized", target_kind="delta")

    delta_train_states = frozen_final_states(
        delta_agent, train_frames, args.batch_size, device)
    delta_test_states = frozen_final_states(
        delta_agent, test_frames, args.batch_size, device)
    delta_reversed_states = frozen_final_states(
        delta_agent, reversed_frames, args.batch_size, device)
    fresh_train_states = frozen_final_states(
        fresh_agent, train_frames, args.batch_size, device)
    fresh_test_states = frozen_final_states(
        fresh_agent, test_frames, args.batch_size, device)
    fresh_reversed_states = frozen_final_states(
        fresh_agent, reversed_frames, args.batch_size, device)

    seed_everything(args.seed + 700)
    initial_action_head = NonlinearActionHead(args.hidden).to(device)
    initial_action_state = copy.deepcopy(initial_action_head.state_dict())
    seed_everything(args.seed + 701)
    initial_value_head = ValueHead(args.hidden).to(device)
    initial_value_state = copy.deepcopy(initial_value_head.state_dict())

    arms: dict[str, dict[str, object]] = {}
    for name in (
            "success_delta", "reinforce_delta",
            "action_shuffled_delta", "success_fresh"):
        arm_started = time.perf_counter()
        action_head = NonlinearActionHead(args.hidden).to(device)
        action_head.load_state_dict(initial_action_state)
        if name == "reinforce_delta":
            value_head = ValueHead(args.hidden).to(device)
            value_head.load_state_dict(initial_value_state)
            learning = train_matched_reinforce(
                action_head, value_head, delta_train_states, train_rules,
                delta_test_states, test_rules,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                epsilon=args.epsilon, seed=args.seed + 100)
            normal_states = delta_test_states
            reversed_states = delta_reversed_states
        else:
            use_fresh = name == "success_fresh"
            learning = train_success_replay(
                action_head,
                fresh_train_states if use_fresh else delta_train_states,
                train_rules,
                fresh_test_states if use_fresh else delta_test_states,
                test_rules,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                epsilon=args.epsilon,
                shuffle_attempted_actions=(
                    name == "action_shuffled_delta"),
                seed=args.seed + 100)
            normal_states = (
                fresh_test_states if use_fresh else delta_test_states)
            reversed_states = (
                fresh_reversed_states if use_fresh
                else delta_reversed_states)
        arms[name] = {
            "learning": learning,
            "reversal_audit": reversal_audit(
                action_head, normal_states, test_rules,
                reversed_states, reversed_rules, args.batch_size),
            "seconds": time.perf_counter() - arm_started,
        }
        print(json.dumps({
            "arm": name,
            **arms[name],
        }, sort_keys=True), flush=True)

    candidate = arms["success_delta"]["learning"]
    control_names = (
        "reinforce_delta", "action_shuffled_delta", "success_fresh")
    controls = [arms[name]["learning"] for name in control_names]
    best_control_aulc = max(
        float(control["reward_aulc_above_chance"])
        for control in controls)
    best_control_final = max(
        float(control["heldout_accuracy_final"])
        for control in controls)
    reversal = arms["success_delta"]["reversal_audit"]
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
        "action_coverage": (
            float(candidate["minimum_action_fraction"]) >= 0.20),
        "reversal_causality": (
            float(reversal["reversed_relabeled_accuracy"]) >= 0.60
            and float(reversal["prediction_flip_rate"]) >= 0.50),
        "advance_to_second_seed": False,
        "advance_to_three_minutes": False,
    }
    gate["advance_to_second_seed"] = bool(
        gate["candidate_beats_controls"]
        and gate["action_coverage"]
        and gate["reversal_causality"])
    report = {
        "schema": "action-conditioned-success-v1",
        "learner_visible": [
            "rendered_rgb_stream", "frozen_recurrent_latent",
            "attempted_action", "logging_propensity",
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
        "representation": representation,
        "arms": arms,
        "gate": gate,
        "total_seconds": time.perf_counter() - started,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "gate": gate,
        "total_seconds": report["total_seconds"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()

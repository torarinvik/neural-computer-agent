"""Test self-supervised action-value prediction on a new contextual primitive.

The controller stays frozen.  The small value probe sees a recurrent state, a
sensory event, and the opaque action that was actually attempted; its sole
training target is the scalar verifier outcome of that attempt.  It never sees
the correct unattempted action or a semantic rule/context label.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from .environment import ACTIONS, NULL_ACTION, CognitiveLifetimeBatch, generate_lifetimes
from .model import UnifiedCognitiveController


class ActionValueProbe(nn.Module):
    def __init__(self, width: int, hidden: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(width * 2 + ACTIONS, hidden), nn.GELU(),
            nn.Linear(hidden, 1))

    def forward(
            self, hidden: torch.Tensor, event: torch.Tensor,
            actions: torch.Tensor) -> torch.Tensor:
        action_one_hot = nn.functional.one_hot(actions, ACTIONS).to(hidden.dtype)
        return self.network(torch.cat([
            hidden, event, action_one_hot], dim=-1)).squeeze(-1)


@torch.no_grad()
def _random_attempts(
        controller: UnifiedCognitiveController, batch: CognitiveLifetimeBatch,
        *, generator: torch.Generator, device: torch.device,
        prediction_start: int = 0,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    state = controller.initial_state(batch.batch_size, device=device)
    previous_action = torch.full(
        (batch.batch_size,), NULL_ACTION, dtype=torch.long, device=device)
    previous_reward = torch.zeros(batch.batch_size, device=device)
    features, attempted, outcomes = [], [], []
    for trial in range(batch.trials):
        feedback = torch.full_like(previous_reward, float(0 < trial <= 2))
        event = controller.vision(batch.frames[:, trial])
        pre_hidden = state.hidden
        _, state = controller.step(
            batch.frames[:, trial], state, previous_action,
            previous_reward * feedback, feedback)
        action = torch.randint(
            ACTIONS, (batch.batch_size,), generator=generator, device="cpu").to(device)
        reward = (action == batch.correct_actions[:, trial]).float()
        if trial >= prediction_start:
            # At a query this is the recurrent state formed from preceding
            # supports plus the current sensory event—the same causal boundary
            # available to an answer adapter before it emits an action.
            features.append(torch.cat([pre_hidden, event], dim=-1))
            attempted.append(action)
            outcomes.append(reward)
        previous_action, previous_reward = action, reward
    return (
        torch.cat(features), torch.cat(attempted), torch.cat(outcomes),
        batch.context_ids.reshape(-1) if batch.context_ids is not None
        else torch.empty(0, device=device, dtype=torch.long))


@torch.no_grad()
def _policy_rollout(
        controller: UnifiedCognitiveController, value: ActionValueProbe,
        batch: CognitiveLifetimeBatch, *, feedback_shuffled: bool = False,
        second_feedback_removed: bool = False,
        blank_vision: bool = False,
        reset_each_trial: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor]:
    state = controller.initial_state(batch.batch_size, device=batch.frames.device)
    previous_action = torch.full(
        (batch.batch_size,), NULL_ACTION, dtype=torch.long, device=batch.frames.device)
    previous_reward = torch.zeros(batch.batch_size, device=batch.frames.device)
    actions, rewards = [], []
    frames = torch.zeros_like(batch.frames) if blank_vision else batch.frames
    for trial in range(batch.trials):
        if reset_each_trial and trial:
            state = controller.initial_state(batch.batch_size, device=batch.frames.device)
        feedback = torch.full_like(previous_reward, float(0 < trial <= 2))
        delivered = previous_reward * feedback
        if trial == 2 and second_feedback_removed:
            feedback = torch.zeros_like(feedback)
            delivered = torch.zeros_like(delivered)
        elif feedback_shuffled and bool(feedback[0]):
            delivered = delivered.roll(1)
        event = controller.vision(frames[:, trial])
        options = torch.arange(ACTIONS, device=frames.device).expand(
            batch.batch_size, -1)
        values = torch.stack([
            value(state.hidden, event, options[:, option])
            for option in range(ACTIONS)], dim=-1)
        action = values.argmax(-1)
        _, state = controller.step(
            frames[:, trial], state, previous_action, delivered, feedback)
        reward = (action == batch.correct_actions[:, trial]).float()
        actions.append(action)
        rewards.append(reward)
        previous_action, previous_reward = action, reward
    return torch.stack(actions, dim=1), torch.stack(rewards, dim=1)


def _metrics(rewards: torch.Tensor, *, query_start: int) -> dict[str, object]:
    accuracy = rewards.mean(dim=0)
    return {
        "accuracy_by_trial": [float(item) for item in accuracy],
        "zero_shot_accuracy": float(accuracy[0]),
        "post_feedback_accuracy": float(accuracy[query_start:].mean()),
    }


@torch.no_grad()
def _evaluate(
        controller: UnifiedCognitiveController, value: ActionValueProbe, *,
        count: int, seed: int, device: torch.device,
        query_start: int,
        ) -> dict[str, object]:
    normal_batch = generate_lifetimes(
        count, 6, seed=seed, heldout=True, task="contextual_mapping",
        support_trials=2, device=device)
    reversed_batch = generate_lifetimes(
        count, 6, seed=seed, heldout=True, task="contextual_mapping",
        support_trials=2, reverse_rules=True, device=device)
    normal_actions, normal_rewards = _policy_rollout(controller, value, normal_batch)
    reversed_actions, reversed_rewards = _policy_rollout(
        controller, value, reversed_batch)
    shuffled_actions, shuffled_rewards = _policy_rollout(
        controller, value, normal_batch, feedback_shuffled=True)
    removed_actions, removed_rewards = _policy_rollout(
        controller, value, normal_batch, second_feedback_removed=True)
    blank_actions, blank_rewards = _policy_rollout(
        controller, value, normal_batch, blank_vision=True)
    reset_actions, reset_rewards = _policy_rollout(
        controller, value, normal_batch, reset_each_trial=True)
    query_contexts = normal_batch.context_ids[:, query_start:]
    assert query_contexts is not None
    normal_post = float(normal_rewards[:, query_start:].mean())
    report = {
        "normal": _metrics(normal_rewards, query_start=query_start),
        "reversed_rule": _metrics(reversed_rewards, query_start=query_start),
        "feedback_shuffled": _metrics(shuffled_rewards, query_start=query_start),
        "second_support_feedback_removed": _metrics(removed_rewards, query_start=query_start),
        "blank_vision": _metrics(blank_rewards, query_start=query_start),
        "active_state_reset": _metrics(reset_rewards, query_start=query_start),
        "normal_query_accuracy_by_context": {
            str(context): float(normal_rewards[:, query_start:][query_contexts == context].mean())
            for context in (0, 1)
        },
        "post_feedback_prediction_flip_rate": float(
            (normal_actions[:, query_start:] != reversed_actions[:, query_start:]).float().mean()),
    }
    report["gate"] = {
        "zero_shot_near_chance": 0.40 <= report["normal"]["zero_shot_accuracy"] <= 0.60,
        "normal_few_shot_at_least_85": normal_post >= 0.85,
        "reversed_few_shot_at_least_85": (
            report["reversed_rule"]["post_feedback_accuracy"] >= 0.85),
        "both_contexts_mastered": all(
            score >= 0.85 for score in report["normal_query_accuracy_by_context"].values()),
        "counterfactual_flip_at_least_80": (
            report["post_feedback_prediction_flip_rate"] >= 0.80),
        "feedback_shuffled_hurts": (
            report["feedback_shuffled"]["post_feedback_accuracy"] <= normal_post - 0.15),
        "second_support_evidence_hurts": (
            report["second_support_feedback_removed"]["post_feedback_accuracy"]
            <= normal_post - 0.15),
        "vision_hurts": (
            report["blank_vision"]["post_feedback_accuracy"] <= normal_post - 0.15),
        "state_hurts": (
            report["active_state_reset"]["post_feedback_accuracy"] <= normal_post - 0.15),
    }
    report["gate"]["accepted"] = all(report["gate"].values())
    return report


def _fit(
        controller: UnifiedCognitiveController, value: ActionValueProbe, *,
        steps: int, batch_size: int, seed: int, device: torch.device,
        replay_epochs: int = 1, prediction_start: int = 0,
        shuffle_outcomes: bool = False,
        ) -> tuple[list[float], int, float]:
    cached_features, cached_actions, cached_outcomes = [], [], []
    for update in range(steps):
        batch = generate_lifetimes(
            batch_size, 6, seed=seed * 10_000 + update,
            task="contextual_mapping", support_trials=2, device=device)
        generator = torch.Generator().manual_seed(seed * 100_000 + update)
        features, actions, outcomes, _ = _random_attempts(
            controller, batch, generator=generator, device=device,
            prediction_start=prediction_start)
        cached_features.append(features)
        cached_actions.append(actions)
        cached_outcomes.append(outcomes)
    features = torch.cat(cached_features)
    actions = torch.cat(cached_actions)
    outcomes = torch.cat(cached_outcomes)
    if shuffle_outcomes:
        outcomes = outcomes.roll(1)
    optimizer = torch.optim.AdamW(value.parameters(), lr=1e-3)
    losses = []
    hidden, event = features[:, :controller.width], features[:, controller.width:]
    for _ in range(replay_epochs):
        loss = nn.functional.binary_cross_entropy_with_logits(
            value(hidden, event, actions), outcomes)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    with torch.no_grad():
        attempted_accuracy = float(
            ((value(hidden, event, actions) >= 0) == outcomes.bool())
            .float().mean())
    return losses, int(outcomes.numel()), attempted_accuracy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument(
        "--replay-epochs", type=int, default=1,
        help=("optimization passes over the same attempted-outcome cache; "
              "does not increase unique verifier interactions"))
    parser.add_argument(
        "--prediction-start", type=int, default=0,
        help=("only predict outcomes at/after this trial; later trials have "
              "more internally available evidence"))
    parser.add_argument(
        "--evaluation-start", type=int, default=2,
        help="first query trial counted by the held-out capability gate")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-lifetimes", type=int, default=512)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    if (
            args.steps < 1 or args.replay_epochs < 1
            or not 0 <= args.prediction_start < 6
            or not 2 <= args.evaluation_start < 6):
        raise ValueError("invalid steps, replay_epochs, or prediction_start")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    controller = UnifiedCognitiveController(
        **payload["model_configuration"]).to(device)
    controller.load_state_dict(payload["state_dict"])
    controller.eval()
    value = ActionValueProbe(controller.width).to(device)
    losses, unique_outcomes, train_attempt_accuracy = _fit(
        controller, value, steps=args.steps, batch_size=args.batch_size,
        seed=args.seed, device=device, replay_epochs=args.replay_epochs,
        prediction_start=args.prediction_start)
    evaluation = _evaluate(
        controller, value, count=args.test_lifetimes,
        seed=args.seed + 9_000_000, device=device,
        query_start=args.evaluation_start)
    shuffled_value = ActionValueProbe(controller.width).to(device)
    shuffled_losses, shuffled_unique_outcomes, shuffled_train_attempt_accuracy = _fit(
        controller, shuffled_value, steps=args.steps, batch_size=args.batch_size,
        seed=args.seed, device=device, replay_epochs=args.replay_epochs,
        prediction_start=args.prediction_start,
        shuffle_outcomes=True)
    shuffled_evaluation = _evaluate(
        controller, shuffled_value, count=args.test_lifetimes,
        seed=args.seed + 9_000_000, device=device,
        query_start=args.evaluation_start)
    report = {
        "schema": "contextual-action-value-probe-v1",
        "claim_boundary": (
            "Frozen-controller action-value diagnostic trained only from "
            "attempted opaque actions and scalar verified outcomes."),
        "semantic_labels_used_for_training": False,
        "unattempted_action_labels_used_for_training": False,
        "checkpoint": str(args.checkpoint),
        "seed": args.seed,
        "steps": args.steps,
        "replay_epochs": args.replay_epochs,
        "prediction_start": args.prediction_start,
        "evaluation_start": args.evaluation_start,
        "batch_size": args.batch_size,
        "unique_verifier_outcomes": unique_outcomes,
        "train_attempt_accuracy": train_attempt_accuracy,
        "shuffled_unique_verifier_outcomes": shuffled_unique_outcomes,
        "shuffled_train_attempt_accuracy": shuffled_train_attempt_accuracy,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "evaluation": evaluation,
        "shuffled_outcome_loss_first": shuffled_losses[0],
        "shuffled_outcome_loss_last": shuffled_losses[-1],
        "shuffled_outcome_evaluation": shuffled_evaluation,
        "shuffled_outcome_control_rejected": not shuffled_evaluation["gate"]["accepted"],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "loss_first": report["loss_first"], "loss_last": report["loss_last"],
        "gate": evaluation["gate"],
        "shuffled_control_rejected": report["shuffled_outcome_control_rejected"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()

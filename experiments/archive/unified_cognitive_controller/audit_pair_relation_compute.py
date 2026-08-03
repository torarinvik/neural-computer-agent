"""Audit how many recurrent controller steps a mastered relation needs.

One controller step is required to encode a sensory event and emit an
intention.  Extra steps replay the same pixels with no action or outcome and
therefore represent optional internal deliberation, not additional evidence.
The verifier remains private and is used only for held-out scoring.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .audit_pair_relation_repertoire import _load
from .environment import (
    NULL_ACTION, CognitiveLifetimeBatch, generate_lifetimes)


@torch.no_grad()
def _rollout_with_extra_thought(
        model, batch: CognitiveLifetimeBatch, *, extra_thought_steps: int,
        feedback_trials: int = 1) -> dict[str, torch.Tensor]:
    if extra_thought_steps < 0:
        raise ValueError("extra thought steps must not be negative")
    state = model.initial_state(
        batch.batch_size, device=batch.frames.device,
        dtype=batch.frames.dtype)
    previous_action = torch.full(
        (batch.batch_size,), NULL_ACTION, device=batch.frames.device,
        dtype=torch.long)
    previous_reward = torch.zeros(
        batch.batch_size, device=batch.frames.device)
    actions, rewards, logits = [], [], []
    null_action = torch.full_like(previous_action, NULL_ACTION)
    zeros = torch.zeros_like(previous_reward)
    for trial in range(batch.trials):
        has_feedback = torch.full_like(
            previous_reward, float(0 < trial <= feedback_trials))
        output, state = model.step(
            batch.frames[:, trial], state, previous_action,
            previous_reward * has_feedback, has_feedback)
        for _ in range(extra_thought_steps):
            output, state = model.step(
                batch.frames[:, trial], state, null_action, zeros, zeros)
        action = output.logits.argmax(-1)
        reward = (
            action == batch.correct_actions[:, trial]).to(
                output.logits.dtype)
        actions.append(action)
        rewards.append(reward)
        logits.append(output.logits)
        previous_action = action
        previous_reward = reward
    return {
        "actions": torch.stack(actions, dim=1),
        "rewards": torch.stack(rewards, dim=1),
        "logits": torch.stack(logits, dim=1),
        "final_workspace": state.workspace,
        "final_hidden": state.hidden,
    }


def _evaluate_budget(
        model, *, count: int, trials: int, seed: int,
        appearance: str, extra_thought_steps: int,
        device: torch.device) -> dict[str, float | int | bool]:
    normal = generate_lifetimes(
        count, trials, seed=seed, heldout=True, task="pair_relation",
        appearance=appearance, support_trials=1, device=device)
    reversed_batch = generate_lifetimes(
        count, trials, seed=seed, heldout=True, task="pair_relation",
        appearance=appearance, support_trials=1, reverse_contexts=True,
        device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    normal_result = _rollout_with_extra_thought(
        model, normal, extra_thought_steps=extra_thought_steps)
    reversed_result = _rollout_with_extra_thought(
        model, reversed_batch, extra_thought_steps=extra_thought_steps)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    normal_accuracy = float(normal_result["rewards"].float().mean())
    reversed_accuracy = float(
        reversed_result["rewards"].float().mean())
    prediction_flip = float(
        (normal_result["actions"] != reversed_result["actions"])
        .float().mean())
    steps_per_lifetime = trials * (1 + extra_thought_steps)
    return {
        "normal_accuracy": normal_accuracy,
        "counterfactual_accuracy": reversed_accuracy,
        "prediction_flip_rate": prediction_flip,
        "extra_thought_steps_per_event": extra_thought_steps,
        "controller_steps_per_lifetime": steps_per_lifetime,
        "logical_lifetimes": count,
        "verifier_bits": count * trials * 2,
        "seconds": elapsed,
        "mastery_at_95": (
            normal_accuracy >= 0.95
            and reversed_accuracy >= 0.95
            and prediction_flip >= 0.80),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=9901)
    parser.add_argument("--count", type=int, default=4096)
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument(
        "--extra-thought-budgets", default="0,1,2,4,8")
    parser.add_argument(
        "--device",
        default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    budgets = tuple(
        int(value) for value in args.extra_thought_budgets.split(","))
    if (
            not budgets or any(value < 0 for value in budgets)
            or len(set(budgets)) != len(budgets)):
        raise ValueError("thought budgets must be unique nonnegative integers")
    if args.count < 2 or args.count % 2:
        raise ValueError("count must be positive and divisible by two")

    device = torch.device(args.device)
    model = _load(args.checkpoint, device)
    results = {}
    for budget in budgets:
        results[str(budget)] = {
            appearance: _evaluate_budget(
                model, count=args.count, trials=args.trials,
                seed=args.seed + 100_000 * budget + 10_000 * index,
                appearance=appearance, extra_thought_steps=budget,
                device=device)
            for index, appearance in enumerate(
                ("bars", "diamonds", "dot_pairs"))
        }
    mastered_budgets = [
        budget for budget in budgets
        if all(
            results[str(budget)][appearance]["mastery_at_95"]
            for appearance in ("bars", "diamonds", "dot_pairs"))]
    minimum = min(mastered_budgets) if mastered_budgets else None
    report = {
        "schema": "pair-relation-compute-audit-v1",
        "claim_boundary": (
            "Each extra thought reprocesses the same learner-visible frame "
            "with no new action, reward, task ID, or verifier information."),
        "checkpoint": str(args.checkpoint),
        "configuration": {
            "seed": args.seed,
            "count": args.count,
            "trials": args.trials,
            "extra_thought_budgets": budgets,
            "device": str(device),
        },
        "results": results,
        "minimum_extra_thought_budget_mastering_all_appearances": minimum,
        "minimum_controller_steps_per_event": (
            1 + minimum if minimum is not None else None),
        "already_compiled_to_physical_minimum": minimum == 0,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "minimum_extra_thought_budget_mastering_all_appearances": minimum,
        "already_compiled_to_physical_minimum": minimum == 0,
        "accuracy": {
            budget: {
                name: round(
                    float(value["normal_accuracy"]), 6)
                for name, value in appearances.items()}
            for budget, appearances in results.items()},
    }, sort_keys=True))


if __name__ == "__main__":
    main()

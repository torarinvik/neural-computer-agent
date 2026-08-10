"""Pressure-test replay-free learned goal progress and verification."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from experiments.external_learned_goal_evaluator import train as learned_goal
from neural_computer import (
    AmodalCognitiveController,
    ExternalGoalEvaluatorStatistics,
)


def _verifier_batch(
    seed: int,
    *,
    shuffled: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed + 100_003)
    train_values = torch.tensor(
        learned_goal.TRAIN_GOALS, dtype=torch.float32
    ) / learned_goal.SCALE
    states: list[torch.Tensor] = []
    goals: list[torch.Tensor] = []
    outcomes: list[float] = []
    near_miss_offsets = (-0.375, -0.25, -0.125, 0.125, 0.25, 0.375)
    for state_index, state_value in enumerate(train_values):
        for goal_index, goal_value in enumerate(train_values):
            for _ in range(learned_goal.VERIFIER_REPEATS):
                if state_index == goal_index:
                    goal_sample = goal_value
                else:
                    offset = near_miss_offsets[
                        (state_index + goal_index + len(outcomes))
                        % len(near_miss_offsets)
                    ]
                    goal_sample = state_value + offset
                states.append(
                    state_value
                    + learned_goal.GOAL_NOISE_STD
                    * torch.randn((), generator=generator)
                )
                goals.append(
                    goal_sample
                    + learned_goal.GOAL_NOISE_STD
                    * torch.randn((), generator=generator)
                )
                distance = abs(float(state_value - goal_sample))
                outcomes.append(
                    float(torch.sigmoid(torch.tensor(6.0 - distance / 0.015))
                    )
                )
    outcome = torch.tensor(outcomes, dtype=torch.float32)
    if shuffled:
        outcome = outcome[torch.randperm(outcome.shape[0], generator=generator)]
    return (
        torch.stack(states).reshape(-1, 1),
        torch.stack(goals).reshape(-1, 1),
        outcome,
    )


def _train_statistics(
    seed: int,
    *,
    shuffled: bool = False,
) -> tuple[ExternalGoalEvaluatorStatistics, dict[str, int | float]]:
    state, goal, outcome = _verifier_batch(seed, shuffled=shuffled)
    evaluator = ExternalGoalEvaluatorStatistics(1, ridge=1e-5)
    evaluator.observe(state, goal, outcome)
    return evaluator, {
        "unique_training_rows": int(state.shape[0]),
        "statistics_updates": 1,
        "replayed_rows": 0,
        "outcome_min": float(outcome.min()),
        "outcome_max": float(outcome.max()),
    }


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    torch.set_num_threads(1)
    controller = AmodalCognitiveController(
        width=8,
        workspace_slots=1,
        intention_width=4,
        feedback_width=2,
        event_window_capacity=2,
    )
    controller_digest = learned_goal._digest(controller)
    for parameter in controller.parameters():
        parameter.requires_grad_(False)

    model, transition_rows = learned_goal._transition_model()
    evaluator, training = _train_statistics(seed)
    evaluator_digest_before_search = evaluator.digest()
    model_digest_before_search = model.digest()
    holdout = learned_goal._verifier_holdout(evaluator, seed)
    learned = learned_goal._evaluate(model, evaluator, seed)
    shuffled = learned_goal._evaluate(model, evaluator, seed, shuffle_goals=True)
    corrupted = learned_goal._evaluate(model, evaluator, seed, corrupt_goals=True)
    shuffled_evaluator, shuffled_training = _train_statistics(
        seed + 500_000, shuffled=True
    )
    reward_shuffled = learned_goal._evaluate(model, shuffled_evaluator, seed)
    restored = ExternalGoalEvaluatorStatistics.from_payload(
        evaluator.state_payload()
    )
    random_floor = learned_goal._random_floor(seed)

    gates = {
        "heldout_verifier_positive": holdout["minimum_positive_probability"] >= 0.8,
        "heldout_verifier_negative": holdout["maximum_negative_probability"] <= 0.2,
        "learned_goal_mastery": learned["mastery"] >= 0.95,
        "beats_goal_shuffle": learned["mastery"] > shuffled["mastery"] + 0.20,
        "beats_reward_shuffle": learned["mastery"] > reward_shuffled["mastery"] + 0.20,
        "corruption_is_not_equivalent": corrupted["mastery"] < 0.95,
        "beats_random_floor": learned["mastery"] > random_floor["mastery"] + 0.20,
        "one_pass_statistics_update": evaluator.sample_count.item()
        == training["unique_training_rows"],
        "evaluator_unchanged_during_search": evaluator.digest()
        == evaluator_digest_before_search,
        "model_unchanged_during_search": model.digest() == model_digest_before_search,
        "exact_evaluator_persistence": restored.digest() == evaluator.digest(),
        "controller_frozen": controller_digest == learned_goal._digest(controller),
    }
    report = {
        "schema": "neural-computer.external-one-pass-goal-evaluator.v1",
        "claim_boundary": (
            "one-pass sufficient-statistics goal verification and planning on "
            "held-out noisy goals; not arbitrary nonlinear goal abstraction, "
            "cross-modal representation migration, or general continual learning"
        ),
        "seed": seed,
        "configuration": {
            "train_goals": list(learned_goal.TRAIN_GOALS),
            "heldout_goals": list(learned_goal.EVAL_GOALS),
            "evaluation_starts": list(learned_goal.EVAL_STARTS),
            "goal_noise_std": learned_goal.GOAL_NOISE_STD,
            "verifier": "graded_scalar_outcomes_one_pass_normal_equations_v1",
            "horizon": learned_goal.HORIZON,
            "beam_width": learned_goal.BEAM_WIDTH,
            "goal_progress_weight": learned_goal.GOAL_PROGRESS_WEIGHT,
            "reward_shuffle_training_seed_offset": 500_000,
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "metrics": {
            "heldout_verifier": holdout,
            "learned_evaluator": learned,
            "goal_shuffled": shuffled,
            "corrupted_goal": corrupted,
            "reward_shuffled_evaluator": reward_shuffled,
            "random_floor": random_floor,
            "transition_model_loss": float(
                model.loss(learned_goal._transition_observation_for_report()).detach()
            ),
            "statistics_training": training,
            "reward_shuffled_training": shuffled_training,
        },
        "accounting": {
            "unique_verifier_outcomes": training["unique_training_rows"],
            "planning_verifier_bits": len(learned_goal.EVAL_GOALS)
            * len(learned_goal.EVAL_STARTS),
            "transition_rows_consumed_once": transition_rows,
            "goal_statistics_updates": training["statistics_updates"],
            "goal_statistics_replayed_rows": training["replayed_rows"],
            "controller_optimizer_updates": 0,
            "old_memory_replay": 0,
            "planner_search_expansions": learned["expanded_nodes"],
            "mean_search_latency_seconds": learned["mean_latency_seconds"],
            "wall_seconds": time.perf_counter() - begun,
        },
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=84201)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()

"""Test a learned opaque goal verifier on held-out noisy goals."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from neural_computer import (
    AmodalCognitiveController,
    ExternalAffineTransitionStatistics,
    ExternalGoalEvaluator,
    ExternalModelBasedPlanner,
    ExternalTransitionObservation,
)

SCALE = 16.0
TRAIN_GOALS = tuple(range(-16, 17, 4))
EVAL_GOALS = tuple(
    position for position in range(-15, 16) if position not in TRAIN_GOALS
)
EVAL_STARTS = (-12, -6, 0, 6, 12)
HORIZON = 32
BEAM_WIDTH = 4
GOAL_PROGRESS_WEIGHT = 1.0
GOAL_NOISE_STD = 0.01
VERIFIER_DISTANCE_SCALE = 0.05
VERIFIER_REPEATS = 8
VERIFIER_UPDATES = 1_000
VERIFIER_HIDDEN_WIDTH = 32
TRANSITION_RIDGE = 1e-5


def _digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        detached = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(detached.dtype).encode("utf-8"))
        digest.update(repr(tuple(detached.shape)).encode("utf-8"))
        digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


def _state(position: int | float) -> torch.Tensor:
    return torch.tensor([[float(position) / SCALE]], dtype=torch.float32)


def _intention(delta: int) -> torch.Tensor:
    return torch.tensor([[float(delta) / SCALE]], dtype=torch.float32)


def _transition_model() -> tuple[ExternalAffineTransitionStatistics, int]:
    states: list[torch.Tensor] = []
    intentions: list[torch.Tensor] = []
    next_states: list[torch.Tensor] = []
    for position in range(-20, 21):
        for delta in (-1, 0, 1):
            states.append(_state(position).squeeze(0))
            intentions.append(_intention(delta).squeeze(0))
            next_states.append(_state(position + delta).squeeze(0))
    observation = ExternalTransitionObservation(
        state=torch.stack(states),
        intention=torch.stack(intentions),
        next_state=torch.stack(next_states),
        confidence=torch.ones(len(states)),
    )
    model = ExternalAffineTransitionStatistics(1, 1, ridge=TRANSITION_RIDGE)
    model.observe(observation)
    return model, int(observation.state.shape[0])


def _verifier_batch(
    seed: int,
    *,
    shuffled: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed + 100_003)
    train_values = torch.tensor(TRAIN_GOALS, dtype=torch.float32) / SCALE
    states: list[torch.Tensor] = []
    goals: list[torch.Tensor] = []
    outcomes: list[float] = []
    for state_index, state_value in enumerate(train_values):
        for goal_index, goal_value in enumerate(train_values):
            for _ in range(VERIFIER_REPEATS):
                is_match = state_index == goal_index
                if is_match:
                    goal_sample = goal_value
                else:
                    near_miss_offsets = (-0.375, -0.25, -0.125, 0.125, 0.25, 0.375)
                    offset = near_miss_offsets[
                        (state_index + goal_index + len(outcomes))
                        % len(near_miss_offsets)
                    ]
                    goal_sample = state_value + offset
                states.append(
                    state_value + GOAL_NOISE_STD * torch.randn((), generator=generator)
                )
                goals.append(
                    goal_sample + GOAL_NOISE_STD * torch.randn((), generator=generator)
                )
                distance = abs(float(state_value - goal_sample))
                outcomes.append(
                    float(torch.exp(torch.tensor(-distance / VERIFIER_DISTANCE_SCALE)))
                )
    outcome = torch.tensor(outcomes, dtype=torch.float32)
    if shuffled:
        permutation = torch.randperm(outcome.shape[0], generator=generator)
        outcome = outcome[permutation]
    return (
        torch.stack(states).reshape(-1, 1),
        torch.stack(goals).reshape(-1, 1),
        outcome,
    )


def _train_evaluator(
    seed: int,
    *,
    shuffled: bool = False,
) -> tuple[ExternalGoalEvaluator, dict[str, int | float]]:
    evaluator = ExternalGoalEvaluator(1, hidden_width=VERIFIER_HIDDEN_WIDTH)
    state, goal, outcome = _verifier_batch(seed, shuffled=shuffled)
    optimizer = torch.optim.Adam(evaluator.parameters(), lr=0.03)
    final_loss = float("inf")
    for _ in range(VERIFIER_UPDATES):
        optimizer.zero_grad()
        loss = evaluator.loss(state, goal, outcome)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
    return evaluator, {
        "unique_training_rows": int(state.shape[0]),
        "optimizer_updates": VERIFIER_UPDATES,
        "repeated_training_rows": int(state.shape[0] * (VERIFIER_UPDATES - 1)),
        "final_loss": final_loss,
    }


def _execute(
    intentions: torch.Tensor,
    start: int,
) -> int:
    position = start
    action_values = torch.tensor([-1.0, 0.0, 1.0]) / SCALE
    for intention in intentions:
        delta = int(torch.argmin((action_values - intention).abs())) - 1
        position += delta
    return position


def _evaluate(
    model: ExternalAffineTransitionStatistics,
    evaluator: ExternalGoalEvaluator,
    seed: int,
    *,
    shuffle_goals: bool = False,
    corrupt_goals: bool = False,
) -> dict[str, object]:
    generator = torch.Generator().manual_seed(seed + 700_009)
    planner = ExternalModelBasedPlanner(
        model,
        beam_width=BEAM_WIDTH,
        goal_evaluator=evaluator,
    )
    successes: list[bool] = []
    latencies: list[float] = []
    expanded_nodes = 0
    for goal_index, goal in enumerate(EVAL_GOALS):
        requested_goal = (
            EVAL_GOALS[-goal_index - 1] if shuffle_goals else goal
        )
        for start in EVAL_STARTS:
            noisy_goal = _state(requested_goal) + GOAL_NOISE_STD * torch.randn(
                1, 1,
                generator=generator,
            )
            if corrupt_goals:
                noisy_goal = noisy_goal + 0.5
            begun = time.perf_counter()
            result = planner.plan(
                _state(start),
                noisy_goal,
                torch.cat((_intention(-1), _intention(0), _intention(1))),
                horizon=HORIZON,
                goal_progress_weight=GOAL_PROGRESS_WEIGHT,
            )
            latencies.append(time.perf_counter() - begun)
            expanded_nodes += result.expanded_nodes
            successes.append(_execute(result.intentions[0], start) == goal)
    return {
        "mastery": sum(successes) / len(successes),
        "successful_trials": sum(successes),
        "trial_count": len(successes),
        "expanded_nodes": expanded_nodes,
        "mean_latency_seconds": sum(latencies) / len(latencies),
    }


def _verifier_holdout(
    evaluator: ExternalGoalEvaluator,
    seed: int,
) -> dict[str, float]:
    generator = torch.Generator().manual_seed(seed + 991_003)
    positives: list[float] = []
    negatives: list[float] = []
    for goal in EVAL_GOALS:
        state = _state(goal) + GOAL_NOISE_STD * torch.randn(1, 1, generator=generator)
        matching_goal = _state(goal) + GOAL_NOISE_STD * torch.randn(
            1, 1,
            generator=generator,
        )
        wrong_goal = _state(goal + 2) + GOAL_NOISE_STD * torch.randn(
            1, 1,
            generator=generator,
        )
        positives.append(float(torch.sigmoid(evaluator(state, matching_goal)).item()))
        negatives.append(float(torch.sigmoid(evaluator(state, wrong_goal)).item()))
    return {
        "minimum_positive_probability": min(positives),
        "maximum_negative_probability": max(negatives),
    }


def _random_floor(seed: int) -> dict[str, float]:
    generator = torch.Generator().manual_seed(seed + 1_003_007)
    successes = 0
    total = 0
    for goal in EVAL_GOALS:
        for start in EVAL_STARTS:
            position = start
            for _ in range(HORIZON):
                position += int(torch.randint(3, (), generator=generator)) - 1
            successes += position == goal
            total += 1
    return {"mastery": successes / total, "trials": total}


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
    controller_digest = _digest(controller)
    for parameter in controller.parameters():
        parameter.requires_grad_(False)

    model, transition_rows = _transition_model()
    evaluator, verifier_training = _train_evaluator(seed)
    evaluator_digest_before_search = evaluator.digest()
    model_digest_before_search = model.digest()
    holdout = _verifier_holdout(evaluator, seed)
    learned = _evaluate(model, evaluator, seed)
    shuffled = _evaluate(model, evaluator, seed, shuffle_goals=True)
    corrupted = _evaluate(model, evaluator, seed, corrupt_goals=True)
    shuffled_evaluator, shuffled_training = _train_evaluator(seed, shuffled=True)
    reward_shuffled = _evaluate(model, shuffled_evaluator, seed)
    restored = ExternalGoalEvaluator.from_payload(evaluator.state_payload())
    random_floor = _random_floor(seed)

    gates = {
        "heldout_verifier_positive": holdout["minimum_positive_probability"] >= 0.8,
        "heldout_verifier_negative": holdout["maximum_negative_probability"] <= 0.2,
        "learned_goal_mastery": learned["mastery"] >= 0.95,
        "beats_goal_shuffle": learned["mastery"] > shuffled["mastery"] + 0.20,
        "beats_reward_shuffle": learned["mastery"] > reward_shuffled["mastery"] + 0.20,
        "corruption_is_not_equivalent": corrupted["mastery"] < 0.95,
        "beats_random_floor": learned["mastery"] > random_floor["mastery"] + 0.20,
        "evaluator_unchanged_during_search": evaluator.digest()
        == evaluator_digest_before_search,
        "model_unchanged_during_search": model.digest() == model_digest_before_search,
        "exact_evaluator_persistence": restored.digest() == evaluator.digest(),
        "controller_frozen": controller_digest == _digest(controller),
    }
    report = {
        "schema": "neural-computer.external-learned-goal-evaluator.v1",
        "claim_boundary": (
            "held-out noisy goal verification and planning through an external "
            "outcome-trained evaluator; not replay-free evaluator learning, "
            "cross-modal goal abstraction, or general continual learning"
        ),
        "seed": seed,
        "configuration": {
            "train_goals": list(TRAIN_GOALS),
            "heldout_goals": list(EVAL_GOALS),
            "evaluation_starts": list(EVAL_STARTS),
            "goal_noise_std": GOAL_NOISE_STD,
            "horizon": HORIZON,
            "beam_width": BEAM_WIDTH,
            "goal_progress_weight": GOAL_PROGRESS_WEIGHT,
            "verifier_training": "graded_scalar_outcomes_with_repeated_offline_batch_v1",
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
            "transition_model_loss": float(model.loss(_transition_observation_for_report()).detach()),
            "verifier_training": verifier_training,
            "reward_shuffled_training": shuffled_training,
        },
        "accounting": {
            "unique_verifier_bits": len(EVAL_GOALS) * len(EVAL_STARTS),
            "goal_verifier_training_rows": verifier_training["unique_training_rows"],
            "goal_verifier_optimizer_updates": verifier_training["optimizer_updates"],
            "goal_verifier_replayed_examples": verifier_training["repeated_training_rows"],
            "transition_rows_consumed_once": transition_rows,
            "external_model_statistics_updates": 1,
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


def _transition_observation_for_report() -> ExternalTransitionObservation:
    states: list[torch.Tensor] = []
    intentions: list[torch.Tensor] = []
    next_states: list[torch.Tensor] = []
    for position in range(-20, 21):
        for delta in (-1, 0, 1):
            states.append(_state(position).squeeze(0))
            intentions.append(_intention(delta).squeeze(0))
            next_states.append(_state(position + delta).squeeze(0))
    return ExternalTransitionObservation(
        state=torch.stack(states),
        intention=torch.stack(intentions),
        next_state=torch.stack(next_states),
        confidence=torch.ones(len(states)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=84101)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()

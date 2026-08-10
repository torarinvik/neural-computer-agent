"""Pressure-test replay-free goal-memory migration across representations."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from experiments.external_learned_goal_evaluator import train as learned_goal
from experiments.external_one_pass_goal_evaluator import train as one_pass
from neural_computer import (
    AmodalCognitiveController,
    ExternalAffineTransitionStatistics,
    ExternalGoalEvaluatorStatistics,
    ExternalGoalRepresentationAlignmentStatistics,
    ExternalModelBasedPlanner,
)

ALIGNMENT_ROWS = 96
ALIGNMENT_NOISE_STD = 0.002
ALIGNMENT_SOURCE_WIDTH = 2
ALIGNMENT_RIDGE = 1e-5
REWARD_SHUFFLE_SEED_OFFSET = 700_001
NEW_TO_OLD_SCALE = 1.7
NEW_TO_OLD_OFFSET = 0.2
NEW_SECOND_SCALE = -0.8
NEW_SECOND_OFFSET = 0.4


def _digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        detached = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(detached.dtype).encode("utf-8"))
        digest.update(repr(tuple(detached.shape)).encode("utf-8"))
        digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


def _new_representation(
    position: float,
    generator: torch.Generator,
) -> torch.Tensor:
    old = float(position) / learned_goal.SCALE
    noise = ALIGNMENT_NOISE_STD * torch.randn(2, generator=generator)
    return torch.tensor(
        [[
            NEW_TO_OLD_SCALE * old + NEW_TO_OLD_OFFSET,
            NEW_SECOND_SCALE * old + NEW_SECOND_OFFSET,
        ]],
        dtype=torch.float32,
    ) + noise.reshape(1, 2)


def _alignment_batch(
    seed: int,
    *,
    shuffled: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed + 300_007)
    positions = torch.linspace(-24.0, 24.0, ALIGNMENT_ROWS)
    source = []
    target = []
    for position in positions:
        source.append(_new_representation(float(position), generator).squeeze(0))
        target.append(torch.tensor([float(position) / learned_goal.SCALE]))
    source_tensor = torch.stack(source)
    target_tensor = torch.stack(target)
    if shuffled:
        target_tensor = target_tensor[
            torch.randperm(target_tensor.shape[0], generator=generator)
        ]
    return source_tensor, target_tensor


def _train_alignment(
    seed: int,
    *,
    shuffled: bool = False,
) -> tuple[ExternalGoalRepresentationAlignmentStatistics, dict[str, int]]:
    source, target = _alignment_batch(seed, shuffled=shuffled)
    adapter = ExternalGoalRepresentationAlignmentStatistics(
        ALIGNMENT_SOURCE_WIDTH,
        1,
        ridge=ALIGNMENT_RIDGE,
    )
    adapter.observe(source, target)
    return adapter, {
        "unique_alignment_pairs": int(source.shape[0]),
        "alignment_statistics_updates": 1,
        "replayed_alignment_pairs": 0,
    }


def _evaluate(
    model: ExternalAffineTransitionStatistics,
    evaluator: ExternalGoalEvaluatorStatistics,
    adapter: ExternalGoalRepresentationAlignmentStatistics,
    seed: int,
    *,
    corrupt_goals: bool = False,
    representation=None,
) -> dict[str, object]:
    if representation is None:
        representation = _new_representation
    generator = torch.Generator().manual_seed(seed + 700_009)
    planner = ExternalModelBasedPlanner(
        model,
        beam_width=learned_goal.BEAM_WIDTH,
        goal_evaluator=evaluator,
    )
    successes: list[bool] = []
    latencies: list[float] = []
    expanded_nodes = 0
    candidates = torch.cat(
        (
            learned_goal._intention(-1),
            learned_goal._intention(0),
            learned_goal._intention(1),
        )
    )
    for goal in learned_goal.EVAL_GOALS:
        for start in learned_goal.EVAL_STARTS:
            new_start = representation(start, generator)
            new_goal = representation(goal, generator)
            if corrupt_goals:
                new_goal = new_goal + torch.tensor([[4.0, 0.0]])
            old_start = adapter(new_start)
            old_goal = adapter(new_goal)
            begun = time.perf_counter()
            result = planner.plan(
                old_start,
                old_goal,
                candidates,
                horizon=learned_goal.HORIZON,
                goal_progress_weight=learned_goal.GOAL_PROGRESS_WEIGHT,
            )
            latencies.append(time.perf_counter() - begun)
            expanded_nodes += result.expanded_nodes
            successes.append(
                learned_goal._execute(result.intentions[0], start) == goal
            )
    return {
        "mastery": sum(successes) / len(successes),
        "successful_trials": sum(successes),
        "trial_count": len(successes),
        "expanded_nodes": expanded_nodes,
        "mean_latency_seconds": sum(latencies) / len(latencies),
    }


def _holdout(
    evaluator: ExternalGoalEvaluatorStatistics,
    adapter: ExternalGoalRepresentationAlignmentStatistics,
    seed: int,
) -> dict[str, float]:
    generator = torch.Generator().manual_seed(seed + 991_003)
    positives: list[float] = []
    negatives: list[float] = []
    for goal in learned_goal.EVAL_GOALS:
        state = adapter(_new_representation(goal, generator))
        matching = adapter(_new_representation(goal, generator))
        wrong = adapter(_new_representation(goal + 2, generator))
        positives.append(float(torch.sigmoid(evaluator(state, matching)).item()))
        negatives.append(float(torch.sigmoid(evaluator(state, wrong)).item()))
    return {
        "minimum_positive_probability": min(positives),
        "maximum_negative_probability": max(negatives),
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
    controller_digest = _digest(controller)
    for parameter in controller.parameters():
        parameter.requires_grad_(False)

    model, transition_rows = learned_goal._transition_model()
    evaluator = ExternalGoalEvaluatorStatistics(1, ridge=1e-5)
    old_state, old_goal, outcome = one_pass._verifier_batch(seed)
    evaluator.observe(old_state, old_goal, outcome)
    evaluator_digest_before_search = evaluator.digest()
    model_digest_before_search = model.digest()
    adapter, alignment_training = _train_alignment(seed)
    adapter_digest_before_search = adapter.digest()
    holdout = _holdout(evaluator, adapter, seed)
    migrated = _evaluate(model, evaluator, adapter, seed)
    corrupted = _evaluate(model, evaluator, adapter, seed, corrupt_goals=True)
    shuffled_adapter, shuffled_alignment_training = _train_alignment(
        seed + 500_000,
        shuffled=True,
    )
    shuffled_alignment = _evaluate(model, evaluator, shuffled_adapter, seed)
    empty_adapter = ExternalGoalRepresentationAlignmentStatistics(
        ALIGNMENT_SOURCE_WIDTH,
        1,
        ridge=ALIGNMENT_RIDGE,
    )
    no_alignment = _evaluate(model, evaluator, empty_adapter, seed)
    reward_shuffled = ExternalGoalEvaluatorStatistics(1, ridge=1e-5)
    reward_state, reward_goal, reward_outcome = one_pass._verifier_batch(
        seed + REWARD_SHUFFLE_SEED_OFFSET,
        shuffled=True,
    )
    reward_shuffled.observe(reward_state, reward_goal, reward_outcome)
    reward_shuffled_eval = _evaluate(model, reward_shuffled, adapter, seed)
    restored_evaluator = ExternalGoalEvaluatorStatistics.from_payload(
        evaluator.state_payload()
    )
    restored_adapter = ExternalGoalRepresentationAlignmentStatistics.from_payload(
        adapter.state_payload()
    )

    gates = {
        "heldout_verifier_positive": holdout["minimum_positive_probability"] >= 0.8,
        "heldout_verifier_negative": holdout["maximum_negative_probability"] <= 0.2,
        "migrated_goal_mastery": migrated["mastery"] >= 0.95,
        "beats_shuffled_alignment": migrated["mastery"]
        > shuffled_alignment["mastery"] + 0.20,
        "beats_missing_alignment": migrated["mastery"]
        > no_alignment["mastery"] + 0.20,
        "beats_reward_shuffle": migrated["mastery"]
        > reward_shuffled_eval["mastery"] + 0.20,
        "corruption_is_not_equivalent": corrupted["mastery"] < 0.95,
        "one_pass_verifier_update": evaluator.sample_count.item() == old_state.shape[0],
        "one_pass_alignment_update": adapter.sample_count.item()
        == alignment_training["unique_alignment_pairs"],
        "evaluator_unchanged_during_search": evaluator.digest()
        == evaluator_digest_before_search,
        "adapter_unchanged_during_search": adapter.digest()
        == adapter_digest_before_search,
        "model_unchanged_during_search": model.digest() == model_digest_before_search,
        "exact_evaluator_persistence": restored_evaluator.digest() == evaluator.digest(),
        "exact_adapter_persistence": restored_adapter.digest() == adapter.digest(),
        "controller_frozen": controller_digest == _digest(controller),
    }
    report = {
        "schema": "neural-computer.external-goal-representation-migration.v1",
        "claim_boundary": (
            "one-pass external goal memory reused through a learned frontend "
            "alignment; not arbitrary nonlinear migration, cross-modal "
            "grounding, or general continual learning"
        ),
        "seed": seed,
        "configuration": {
            "old_representation_width": 1,
            "new_representation_width": ALIGNMENT_SOURCE_WIDTH,
            "alignment": "one-pass_affine_new_to_old_with_bias_v1",
            "alignment_rows": ALIGNMENT_ROWS,
            "alignment_noise_std": ALIGNMENT_NOISE_STD,
            "old_verifier": "one_pass_graded_goal_statistics_v2",
            "horizon": learned_goal.HORIZON,
            "beam_width": learned_goal.BEAM_WIDTH,
            "goal_progress_weight": learned_goal.GOAL_PROGRESS_WEIGHT,
            "reward_shuffle_seed_offset": REWARD_SHUFFLE_SEED_OFFSET,
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "metrics": {
            "heldout_verifier": holdout,
            "migrated_goal_mastery": migrated,
            "shuffled_alignment": shuffled_alignment,
            "missing_alignment": no_alignment,
            "reward_shuffled_evaluator": reward_shuffled_eval,
            "corrupted_goal": corrupted,
            "statistics_training": {
                "unique_verifier_outcomes": int(old_state.shape[0]),
                "verifier_statistics_updates": 1,
                "verifier_replayed_rows": 0,
                "outcome_min": float(outcome.min()),
                "outcome_max": float(outcome.max()),
            },
            "alignment_training": alignment_training,
            "shuffled_alignment_training": shuffled_alignment_training,
            "reward_shuffled_outcomes": int(reward_outcome.shape[0]),
        },
        "accounting": {
            "unique_verifier_outcomes": int(old_state.shape[0]),
            "unique_alignment_pairs": alignment_training["unique_alignment_pairs"],
            "transition_rows_consumed_once": transition_rows,
            "verifier_statistics_updates": 1,
            "alignment_statistics_updates": 1,
            "old_verifier_replay": 0,
            "old_alignment_replay": 0,
            "controller_optimizer_updates": 0,
            "planner_search_expansions": migrated["expanded_nodes"],
            "mean_search_latency_seconds": migrated["mean_latency_seconds"],
            "wall_seconds": time.perf_counter() - begun,
        },
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=84301)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()

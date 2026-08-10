"""Pressure-test replay-free nonlinear goal representation alignment."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from experiments.external_goal_representation_drift_gate import train as drift
from experiments.external_goal_representation_migration import train as migration
from experiments.external_one_pass_goal_evaluator import train as one_pass
from neural_computer import (
    AmodalCognitiveController,
    ExternalGoalEvaluatorStatistics,
    ExternalGoalRepresentationAlignmentStatistics,
    ExternalGoalRepresentationRandomFeatureAlignmentStatistics,
)

FEATURE_WIDTH = 64
FEATURE_SEED_OFFSET = 9_001
TRAIN_STRIDE = 2
ALIGNMENT_TOLERANCE = 5e-3
SHUFFLE_SEED_OFFSET = 500_000


def _fit_random(
    source: torch.Tensor,
    indices: torch.Tensor,
    training_target: torch.Tensor,
    seed: int,
) -> ExternalGoalRepresentationRandomFeatureAlignmentStatistics:
    adapter = ExternalGoalRepresentationRandomFeatureAlignmentStatistics(
        2,
        1,
        feature_width=FEATURE_WIDTH,
        ridge=1e-4,
        seed=seed + FEATURE_SEED_OFFSET,
    )
    adapter.observe(source.index_select(0, indices), training_target)
    return adapter


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
    controller_digest = migration._digest(controller)
    for parameter in controller.parameters():
        parameter.requires_grad_(False)

    model, transition_rows = migration.learned_goal._transition_model()
    evaluator = ExternalGoalEvaluatorStatistics(1, ridge=1e-5)
    old_state, old_goal, outcome = one_pass._verifier_batch(seed)
    evaluator.observe(old_state, old_goal, outcome)
    evaluator_digest_before = evaluator.digest()
    model_digest_before = model.digest()

    source, target = drift._nonlinear_batch(seed)
    train_indices = torch.arange(0, source.shape[0], TRAIN_STRIDE)
    heldout_indices = torch.arange(1, source.shape[0], TRAIN_STRIDE)
    nonlinear_adapter = _fit_random(
        source,
        train_indices,
        target.index_select(0, train_indices),
        seed,
    )
    nonlinear_receipt = nonlinear_adapter.verify_heldout(
        source.index_select(0, heldout_indices),
        target.index_select(0, heldout_indices),
        prediction_tolerance=ALIGNMENT_TOLERANCE,
    )
    nonlinear_adapter_digest_before_search = nonlinear_adapter.digest()
    migrated = migration._evaluate(
        model,
        evaluator,
        nonlinear_adapter,
        seed,
        representation=drift._nonlinear_representation,
    )

    linear_adapter = ExternalGoalRepresentationAlignmentStatistics(2, 1, ridge=1e-5)
    linear_adapter.observe(
        source.index_select(0, train_indices), target.index_select(0, train_indices)
    )
    linear_receipt = linear_adapter.verify_heldout(
        source.index_select(0, heldout_indices),
        target.index_select(0, heldout_indices),
        prediction_tolerance=ALIGNMENT_TOLERANCE,
    )

    generator = torch.Generator().manual_seed(seed + SHUFFLE_SEED_OFFSET)
    permutation = torch.randperm(train_indices.numel(), generator=generator)
    shuffled_adapter = _fit_random(
        source,
        train_indices,
        target.index_select(0, train_indices).index_select(0, permutation),
        seed + SHUFFLE_SEED_OFFSET,
    )
    shuffled = migration._evaluate(
        model,
        evaluator,
        shuffled_adapter,
        seed,
        representation=drift._nonlinear_representation,
    )
    restored = ExternalGoalRepresentationRandomFeatureAlignmentStatistics.from_payload(
        nonlinear_adapter.state_payload()
    )

    gates = {
        "nonlinear_alignment_accepted": nonlinear_receipt.accepted,
        "nonlinear_goal_mastery": migrated["mastery"] >= 0.95,
        "linear_candidate_rejected": not linear_receipt.accepted,
        "beats_shuffled_alignment": migrated["mastery"] > shuffled["mastery"] + 0.20,
        "adapter_unchanged_during_search": nonlinear_adapter.digest()
        == nonlinear_adapter_digest_before_search,
        "verifier_memory_unchanged": evaluator.digest() == evaluator_digest_before,
        "model_unchanged": model.digest() == model_digest_before,
        "exact_adapter_persistence": restored.digest() == nonlinear_adapter.digest(),
        "controller_frozen": controller_digest == migration._digest(controller),
        "verifier_replay_zero": evaluator.sample_count.item() == outcome.shape[0],
        "one_pass_alignment_update": nonlinear_adapter.sample_count.item()
        == train_indices.numel(),
    }
    report = {
        "schema": "neural-computer.external-goal-representation-nonlinear-alignment.v1",
        "claim_boundary": (
            "bounded replay-free random-feature nonlinear goal alignment across "
            "a changed frontend; not arbitrary nonlinear computation or general continual learning"
        ),
        "seed": seed,
        "configuration": {
            "representation": "old_scalar_to_new_squared_cubic_v1",
            "feature_width": FEATURE_WIDTH,
            "feature_seed_offset": FEATURE_SEED_OFFSET,
            "train_alignment_pairs": int(train_indices.numel()),
            "heldout_alignment_pairs": int(heldout_indices.numel()),
            "alignment_tolerance": ALIGNMENT_TOLERANCE,
            "horizon": migration.learned_goal.HORIZON,
            "beam_width": migration.learned_goal.BEAM_WIDTH,
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "metrics": {
            "nonlinear_alignment_receipt": nonlinear_receipt.__dict__,
            "linear_alignment_receipt": linear_receipt.__dict__,
            "migrated_goal_mastery": migrated,
            "shuffled_alignment": shuffled,
            "verifier_training": {
                "unique_verifier_outcomes": int(outcome.shape[0]),
                "statistics_updates": 1,
                "replayed_rows": 0,
            },
        },
        "accounting": {
            "unique_verifier_outcomes": int(outcome.shape[0]),
            "unique_alignment_pairs": int(train_indices.numel()),
            "heldout_alignment_pairs": int(heldout_indices.numel()),
            "transition_rows_consumed_once": transition_rows,
            "verifier_statistics_updates": 1,
            "alignment_statistics_updates": 1,
            "old_verifier_replay": 0,
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
    parser.add_argument("--seed", type=int, default=84501)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()

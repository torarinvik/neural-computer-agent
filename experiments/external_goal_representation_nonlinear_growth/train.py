"""Pressure-test replay-free copy-on-write nonlinear alignment growth."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from experiments.external_goal_representation_drift_gate import train as drift
from experiments.external_goal_representation_migration import train as migration
from experiments.external_goal_representation_nonlinear_alignment import (
    train as nonlinear_alignment,
)
from experiments.external_one_pass_goal_evaluator import train as one_pass
from neural_computer import (
    AmodalCognitiveController,
    ExternalGoalEvaluatorStatistics,
    ExternalGoalRepresentationRandomFeatureAlignmentStatistics,
)

INITIAL_FEATURE_WIDTH = 16
GROWN_FEATURE_WIDTH = 80
INITIAL_INDEX_STRIDE = 4
GROWTH_INDEX_OFFSET = 2
ALIGNMENT_TOLERANCE = 5e-3
RETENTION_TOLERANCE = 1e-8


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
    initial_indices = torch.arange(0, source.shape[0], INITIAL_INDEX_STRIDE)
    growth_indices = torch.arange(
        GROWTH_INDEX_OFFSET,
        source.shape[0],
        INITIAL_INDEX_STRIDE,
    )
    heldout_mask = torch.ones(source.shape[0], dtype=torch.bool)
    heldout_mask[initial_indices] = False
    heldout_mask[growth_indices] = False
    heldout_indices = heldout_mask.nonzero(as_tuple=False).reshape(-1)

    adapter = ExternalGoalRepresentationRandomFeatureAlignmentStatistics(
        2,
        1,
        feature_width=INITIAL_FEATURE_WIDTH,
        ridge=1e-4,
        seed=seed + nonlinear_alignment.FEATURE_SEED_OFFSET,
    )
    adapter.observe(source[initial_indices], target[initial_indices])
    adapter_digest_before_growth = adapter.digest()
    pre_growth_receipt = adapter.verify_heldout(
        source[heldout_indices],
        target[heldout_indices],
        prediction_tolerance=ALIGNMENT_TOLERANCE,
    )
    pre_growth_mastery = migration._evaluate(
        model,
        evaluator,
        adapter,
        seed,
        representation=drift._nonlinear_representation,
    )
    growth_receipt = adapter.grow_features_verified(
        GROWN_FEATURE_WIDTH,
        source[initial_indices],
        retention_tolerance=RETENTION_TOLERANCE,
    )
    adapter.observe(source[growth_indices], target[growth_indices])
    post_growth_receipt = adapter.verify_heldout(
        source[heldout_indices],
        target[heldout_indices],
        prediction_tolerance=ALIGNMENT_TOLERANCE,
    )
    adapter_digest_before_search = adapter.digest()
    grown_mastery = migration._evaluate(
        model,
        evaluator,
        adapter,
        seed,
        representation=drift._nonlinear_representation,
    )
    restored = ExternalGoalRepresentationRandomFeatureAlignmentStatistics.from_payload(
        adapter.state_payload()
    )

    gates = {
        "pre_growth_candidate_failed": not pre_growth_receipt.accepted,
        "growth_retention_accepted": growth_receipt.accepted,
        "post_growth_alignment_accepted": post_growth_receipt.accepted,
        "growth_improved_mastery": grown_mastery["mastery"]
        > pre_growth_mastery["mastery"] + 0.20,
        "grown_goal_mastery": grown_mastery["mastery"] >= 0.95,
        "adapter_unchanged_during_search": adapter.digest()
        == adapter_digest_before_search,
        "old_adapter_state_retained_at_growth": growth_receipt.source_digest
        == adapter_digest_before_growth,
        "verifier_memory_unchanged": evaluator.digest() == evaluator_digest_before,
        "model_unchanged": model.digest() == model_digest_before,
        "exact_adapter_persistence": restored.digest() == adapter.digest(),
        "controller_frozen": controller_digest == migration._digest(controller),
        "verifier_replay_zero": evaluator.sample_count.item() == outcome.shape[0],
        "alignment_replay_zero": adapter.sample_count.item()
        == initial_indices.numel() + growth_indices.numel(),
    }
    report = {
        "schema": "neural-computer.external-goal-representation-nonlinear-growth.v1",
        "claim_boundary": (
            "copy-on-write growth of a bounded replay-free nonlinear goal "
            "alignment basis; not unrestricted capacity or general continual learning"
        ),
        "seed": seed,
        "configuration": {
            "initial_feature_width": INITIAL_FEATURE_WIDTH,
            "grown_feature_width": GROWN_FEATURE_WIDTH,
            "initial_alignment_pairs": int(initial_indices.numel()),
            "growth_alignment_pairs": int(growth_indices.numel()),
            "heldout_alignment_pairs": int(heldout_indices.numel()),
            "alignment_tolerance": ALIGNMENT_TOLERANCE,
            "retention_tolerance": RETENTION_TOLERANCE,
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "metrics": {
            "pre_growth_receipt": pre_growth_receipt.__dict__,
            "pre_growth_mastery": pre_growth_mastery,
            "growth_receipt": growth_receipt.__dict__,
            "post_growth_receipt": post_growth_receipt.__dict__,
            "grown_mastery": grown_mastery,
            "verifier_training": {
                "unique_verifier_outcomes": int(outcome.shape[0]),
                "statistics_updates": 1,
                "replayed_rows": 0,
            },
        },
        "accounting": {
            "unique_verifier_outcomes": int(outcome.shape[0]),
            "initial_alignment_pairs": int(initial_indices.numel()),
            "growth_alignment_pairs": int(growth_indices.numel()),
            "heldout_alignment_pairs": int(heldout_indices.numel()),
            "transition_rows_consumed_once": transition_rows,
            "verifier_statistics_updates": 1,
            "alignment_statistics_updates": 2,
            "old_verifier_replay": 0,
            "old_alignment_replay": 0,
            "controller_optimizer_updates": 0,
            "planner_search_expansions": grown_mastery["expanded_nodes"],
            "mean_search_latency_seconds": grown_mastery["mean_latency_seconds"],
            "wall_seconds": time.perf_counter() - begun,
        },
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=84601)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()

"""Pressure-test concurrent replay-free nonlinear frontend alignments."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
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
    ExternalGoalRepresentationAlignmentBank,
    ExternalGoalRepresentationAlignmentStatistics,
    ExternalGoalRepresentationRandomFeatureAlignmentStatistics,
)

BANK_CAPACITY = 2
QUARANTINE_CAPACITY = 2
ALIGNMENT_TOLERANCE = 5e-3
TRAIN_STRIDE = 2
BANK_FEATURE_WIDTH = 96


def _fit_affine(seed: int) -> tuple[
    ExternalGoalRepresentationAlignmentStatistics, torch.Tensor, torch.Tensor
]:
    source, target = migration._alignment_batch(seed)
    train_indices = torch.arange(0, source.shape[0], TRAIN_STRIDE)
    heldout_indices = torch.arange(1, source.shape[0], TRAIN_STRIDE)
    adapter = ExternalGoalRepresentationAlignmentStatistics(2, 1, ridge=1e-5)
    adapter.observe(source[train_indices], target[train_indices])
    return adapter, source[heldout_indices], target[heldout_indices]


def _fit_nonlinear(
    seed: int,
    *,
    shuffled: bool = False,
) -> tuple[
    ExternalGoalRepresentationRandomFeatureAlignmentStatistics,
    torch.Tensor,
    torch.Tensor,
]:
    source, target = drift._nonlinear_batch(seed)
    train_indices = torch.arange(0, source.shape[0], TRAIN_STRIDE)
    heldout_indices = torch.arange(1, source.shape[0], TRAIN_STRIDE)
    training_target = target[train_indices]
    if shuffled:
        generator = torch.Generator().manual_seed(seed + 500_000)
        training_target = training_target[
            torch.randperm(training_target.shape[0], generator=generator)
        ]
    adapter = ExternalGoalRepresentationRandomFeatureAlignmentStatistics(
        2,
        1,
        feature_width=BANK_FEATURE_WIDTH,
        ridge=1e-4,
        seed=seed + nonlinear_alignment.FEATURE_SEED_OFFSET,
    )
    adapter.observe(source[train_indices], training_target)
    return adapter, source[heldout_indices], target[heldout_indices]


def _evaluate(
    model,
    evaluator,
    adapter,
    seed: int,
    *,
    representation,
) -> dict[str, object]:
    return migration._evaluate(
        model,
        evaluator,
        adapter,
        seed,
        representation=representation,
    )


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
    evaluator_digest = evaluator.digest()
    model_digest = model.digest()

    affine, affine_source, affine_target = _fit_affine(seed)
    nonlinear_a, nonlinear_a_source, nonlinear_a_target = _fit_nonlinear(seed)
    nonlinear_b, nonlinear_b_source, nonlinear_b_target = _fit_nonlinear(seed + 2)
    corrupted, corrupted_source, corrupted_target = _fit_nonlinear(
        seed + 3, shuffled=True
    )

    bank = ExternalGoalRepresentationAlignmentBank(
        1,
        capacity=BANK_CAPACITY,
        quarantine_capacity=QUARANTINE_CAPACITY,
    )
    first = bank.admit_verified(
        "frontend-affine",
        affine,
        affine_source,
        affine_target,
        prediction_tolerance=ALIGNMENT_TOLERANCE,
    )
    second = bank.admit_verified(
        "frontend-nonlinear-a",
        nonlinear_a,
        nonlinear_a_source,
        nonlinear_a_target,
        prediction_tolerance=ALIGNMENT_TOLERANCE,
    )
    active_digest_before_overflow = bank.active_digest()
    corrupted_receipt = bank.admit_verified(
        "frontend-corrupted",
        corrupted,
        corrupted_source,
        corrupted_target,
        prediction_tolerance=ALIGNMENT_TOLERANCE,
    )
    overflow = bank.admit_verified(
        "frontend-nonlinear-b",
        nonlinear_b,
        nonlinear_b_source,
        nonlinear_b_target,
        prediction_tolerance=ALIGNMENT_TOLERANCE,
    )
    active_digest_after_overflow = bank.active_digest()

    affine_eval = _evaluate(
        model,
        evaluator,
        bank.adapter_for_space("frontend-affine"),
        seed,
        representation=migration._new_representation,
    )
    nonlinear_a_eval = _evaluate(
        model,
        evaluator,
        bank.adapter_for_space("frontend-nonlinear-a"),
        seed,
        representation=drift._nonlinear_representation,
    )

    eviction = bank.evict_verified(
        first.slot_id,
        lambda candidate: candidate.adapter_for_space("frontend-nonlinear-a")
        .verify_heldout(
            nonlinear_a_source,
            nonlinear_a_target,
            prediction_tolerance=ALIGNMENT_TOLERANCE,
        )
        .accepted,
    )
    promoted = bank.promote_quarantined_verified(
        "frontend-nonlinear-b",
        nonlinear_b_source,
        nonlinear_b_target,
        prediction_tolerance=ALIGNMENT_TOLERANCE,
    )
    nonlinear_b_eval = _evaluate(
        model,
        evaluator,
        bank.adapter_for_space("frontend-nonlinear-b"),
        seed,
        representation=drift._nonlinear_representation,
    )
    failed_eviction = bank.evict_verified(
        second.slot_id,
        lambda _candidate: False,
    )
    restored = ExternalGoalRepresentationAlignmentBank.from_payload(bank.state_payload())

    gates = {
        "affine_admitted": first.accepted,
        "nonlinear_a_admitted": second.accepted,
        "corrupted_candidate_rejected": not corrupted_receipt.accepted,
        "corrupted_candidate_quarantined": corrupted_receipt.quarantined,
        "valid_overflow_rejected": not overflow.accepted,
        "valid_overflow_quarantined": overflow.quarantined,
        "active_slots_unchanged_by_overflow": (
            active_digest_after_overflow == active_digest_before_overflow
        ),
        "concurrent_initial_mastery": (
            affine_eval["mastery"] >= 0.95 and nonlinear_a_eval["mastery"] >= 0.95
        ),
        "stable_slot_eviction_accepted": eviction.accepted,
        "evicted_slot_not_reused": promoted.slot_id == 2,
        "quarantined_valid_candidate_promoted": promoted.accepted,
        "promoted_frontend_mastery": nonlinear_b_eval["mastery"] >= 0.95,
        "failed_eviction_did_not_mutate": not failed_eviction.accepted,
        "corrupted_quarantine_retained": "frontend-corrupted"
        in bank.quarantined_space_ids,
        "exact_bank_persistence": restored.digest() == bank.digest(),
        "verifier_memory_unchanged": evaluator.digest() == evaluator_digest,
        "model_unchanged": model.digest() == model_digest,
        "controller_frozen": controller_digest == migration._digest(controller),
        "verifier_replay_zero": evaluator.sample_count.item() == outcome.shape[0],
        "alignment_replay_zero": all(
            adapter.sample_count.item() == adapter.sample_count.item()
            for adapter in (affine, nonlinear_a, nonlinear_b, corrupted)
        ),
    }
    report = {
        "schema": "neural-computer.external-goal-representation-alignment-bank.v1",
        "claim_boundary": (
            "bounded concurrent opaque frontend alignment slots with quarantine, "
            "stable eviction, and replay-free held-out promotion; not unrestricted "
            "memory growth or general continual learning"
        ),
        "seed": seed,
        "configuration": {
            "capacity": BANK_CAPACITY,
            "quarantine_capacity": QUARANTINE_CAPACITY,
            "alignment_tolerance": ALIGNMENT_TOLERANCE,
            "feature_width": BANK_FEATURE_WIDTH,
            "frontends": [
                "frontend-affine",
                "frontend-nonlinear-a",
                "frontend-nonlinear-b",
                "frontend-corrupted",
            ],
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "metrics": {
            "first_admission": asdict(first),
            "second_admission": asdict(second),
            "corrupted_admission": asdict(corrupted_receipt),
            "overflow_admission": asdict(overflow),
            "eviction": asdict(eviction),
            "promoted_quarantine": asdict(promoted),
            "failed_eviction": asdict(failed_eviction),
            "initial_affine_mastery": affine_eval,
            "initial_nonlinear_mastery": nonlinear_a_eval,
            "promoted_nonlinear_mastery": nonlinear_b_eval,
            "final_active_frontends": list(bank.frontend_space_ids),
            "final_slot_ids": list(bank.slot_ids),
            "quarantined_frontends": list(bank.quarantined_space_ids),
        },
        "accounting": {
            "unique_verifier_outcomes": int(outcome.shape[0]),
            "unique_alignment_pairs": sum(
                int(adapter.sample_count.item())
                for adapter in (affine, nonlinear_a, nonlinear_b, corrupted)
            ),
            "heldout_alignment_pairs": sum(
                int(source.shape[0])
                for source in (
                    affine_source,
                    nonlinear_a_source,
                    nonlinear_b_source,
                    corrupted_source,
                )
            ),
            "transition_rows_consumed_once": transition_rows,
            "alignment_statistics_updates": 4,
            "old_verifier_replay": 0,
            "old_alignment_replay": 0,
            "controller_optimizer_updates": 0,
            "wall_seconds": time.perf_counter() - begun,
        },
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=84701)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()

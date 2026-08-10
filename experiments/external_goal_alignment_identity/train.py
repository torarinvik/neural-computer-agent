"""Pressure-test learned opaque identity routing for goal alignments."""

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
IDENTITY_WIDTH = 8
IDENTITY_MIN_SCORE = 0.60
IDENTITY_MIN_MARGIN = 0.20
ALIGNMENT_TOLERANCE = 5e-3
TRAIN_STRIDE = 2
FEATURE_WIDTH = 96


def _signature(source: torch.Tensor) -> torch.Tensor:
    """Build a generic learned-event summary, not a frontend label."""

    values = source.to(dtype=torch.float32)
    summary = torch.cat(
        (
            values.mean(dim=0),
            values.std(dim=0),
            values.amin(dim=0),
            values.amax(dim=0),
        )
    )
    return torch.nn.functional.normalize(summary, dim=0)


def _split(source: torch.Tensor, target: torch.Tensor) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    train_indices = torch.arange(0, source.shape[0], TRAIN_STRIDE)
    heldout_indices = torch.arange(1, source.shape[0], TRAIN_STRIDE)
    return (
        source[train_indices],
        target[train_indices],
        source[heldout_indices],
        target[heldout_indices],
        train_indices,
    )


def _fit_affine(
    seed: int,
) -> tuple[ExternalGoalRepresentationAlignmentStatistics, torch.Tensor, torch.Tensor, torch.Tensor]:
    source, target = migration._alignment_batch(seed)
    train_source, train_target, heldout_source, heldout_target, _ = _split(source, target)
    adapter = ExternalGoalRepresentationAlignmentStatistics(2, 1, ridge=1e-5)
    adapter.observe(train_source, train_target)
    return adapter, heldout_source, heldout_target, _signature(train_source)


def _fit_nonlinear(
    source: torch.Tensor,
    target: torch.Tensor,
    seed: int,
    *,
    shuffled: bool = False,
) -> tuple[
    ExternalGoalRepresentationRandomFeatureAlignmentStatistics,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    train_source, train_target, heldout_source, heldout_target, _ = _split(source, target)
    if shuffled:
        generator = torch.Generator().manual_seed(seed + 500_000)
        train_target = train_target[
            torch.randperm(train_target.shape[0], generator=generator)
        ]
    adapter = ExternalGoalRepresentationRandomFeatureAlignmentStatistics(
        2,
        1,
        feature_width=FEATURE_WIDTH,
        ridge=1e-4,
        seed=seed + nonlinear_alignment.FEATURE_SEED_OFFSET,
    )
    adapter.observe(train_source, train_target)
    return adapter, heldout_source, heldout_target, _signature(train_source)


def _swapped_batch(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    source, target = drift._nonlinear_batch(seed)
    return source.flip(1), target


def _swapped_representation(
    position: float,
    generator: torch.Generator,
) -> torch.Tensor:
    return drift._nonlinear_representation(position, generator).flip(1)


def _passive_source(
    representation,
    seed: int,
    *,
    count: int = 32,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed + 910_007)
    positions = torch.linspace(-24.0, 24.0, count)
    return torch.stack(
        [representation(float(position), generator).squeeze(0) for position in positions]
    )


def _route_accuracy(
    bank: ExternalGoalRepresentationAlignmentBank,
    queries: list[tuple[object, int]],
    seed: int,
) -> tuple[float, list[int | None]]:
    selected: list[int | None] = []
    for index, (representation, expected_slot) in enumerate(queries):
        passive = _passive_source(representation, seed + index)
        result = bank.route_by_signature(_signature(passive), passive[:1])
        selected.append(result.selected_slot_id)
    correct = sum(
        choice == expected for choice, (_, expected) in zip(selected, queries, strict=True)
    )
    return correct / len(queries), selected


def _evaluate_routed(
    bank: ExternalGoalRepresentationAlignmentBank,
    model,
    evaluator,
    representation,
    seed: int,
    *,
    query_expected_slot: int,
) -> tuple[dict[str, object], object]:
    passive = _passive_source(representation, seed)
    route = bank.route_by_signature(_signature(passive), passive[:1])
    if route.selected_slot_id != query_expected_slot:
        return {
            "mastery": 0.0,
            "successful_trials": 0,
            "trial_count": len(migration.learned_goal.EVAL_GOALS)
            * len(migration.learned_goal.EVAL_STARTS),
            "expanded_nodes": 0,
            "mean_latency_seconds": 0.0,
        }, route
    generator = torch.Generator().manual_seed(seed + 700_009)
    planner = migration.ExternalModelBasedPlanner(
        model,
        beam_width=migration.learned_goal.BEAM_WIDTH,
        goal_evaluator=evaluator,
    )
    candidates = torch.cat(
        (
            migration.learned_goal._intention(-1),
            migration.learned_goal._intention(0),
            migration.learned_goal._intention(1),
        )
    )
    successes: list[bool] = []
    latencies: list[float] = []
    expanded_nodes = 0
    for goal in migration.learned_goal.EVAL_GOALS:
        for start in migration.learned_goal.EVAL_STARTS:
            new_start = representation(start, generator)
            new_goal = representation(goal, generator)
            old_start = bank.route_slot(route.selected_slot_id, new_start)
            old_goal = bank.route_slot(route.selected_slot_id, new_goal)
            begun = time.perf_counter()
            result = planner.plan(
                old_start,
                old_goal,
                candidates,
                horizon=migration.learned_goal.HORIZON,
                goal_progress_weight=migration.learned_goal.GOAL_PROGRESS_WEIGHT,
            )
            latencies.append(time.perf_counter() - begun)
            expanded_nodes += result.expanded_nodes
            successes.append(
                migration.learned_goal._execute(result.intentions[0], start) == goal
            )
    return {
        "mastery": sum(successes) / len(successes),
        "successful_trials": sum(successes),
        "trial_count": len(successes),
        "expanded_nodes": expanded_nodes,
        "mean_latency_seconds": sum(latencies) / len(latencies),
    }, route


def _route_summary(result) -> dict[str, object]:
    return {
        "selected_slot_id": result.selected_slot_id,
        "eligible_slot_ids": list(result.eligible_slot_ids),
        "scores": [float(value) for value in result.scores.tolist()],
        "margin": result.margin,
        "reason": result.reason,
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
    controller_digest = migration._digest(controller)
    for parameter in controller.parameters():
        parameter.requires_grad_(False)

    model, transition_rows = migration.learned_goal._transition_model()
    evaluator = ExternalGoalEvaluatorStatistics(1, ridge=1e-5)
    old_state, old_goal, outcome = one_pass._verifier_batch(seed)
    evaluator.observe(old_state, old_goal, outcome)
    evaluator_digest = evaluator.digest()
    model_digest = model.digest()

    affine, affine_source, affine_target, affine_signature = _fit_affine(seed)
    nonlinear_source, nonlinear_target = drift._nonlinear_batch(seed)
    nonlinear, nonlinear_holdout_source, nonlinear_holdout_target, nonlinear_signature = _fit_nonlinear(
        nonlinear_source, nonlinear_target, seed
    )
    swapped_source, swapped_target = _swapped_batch(seed + 1)
    swapped, swapped_holdout_source, swapped_holdout_target, swapped_signature = _fit_nonlinear(
        swapped_source, swapped_target, seed + 1
    )
    corrupted, corrupted_holdout_source, corrupted_holdout_target, corrupted_signature = _fit_nonlinear(
        nonlinear_source, nonlinear_target, seed + 2, shuffled=True
    )

    bank = ExternalGoalRepresentationAlignmentBank(
        1,
        capacity=BANK_CAPACITY,
        quarantine_capacity=QUARANTINE_CAPACITY,
        identity_width=IDENTITY_WIDTH,
        identity_min_score=IDENTITY_MIN_SCORE,
        identity_min_margin=IDENTITY_MIN_MARGIN,
    )
    # Admission order is intentionally different from the later runtime order.
    admitted_nonlinear = bank.admit_verified(
        "opaque-nonlinear",
        nonlinear,
        nonlinear_holdout_source,
        nonlinear_holdout_target,
        prediction_tolerance=ALIGNMENT_TOLERANCE,
        identity_signature=nonlinear_signature,
    )
    admitted_affine = bank.admit_verified(
        "opaque-affine",
        affine,
        affine_source,
        affine_target,
        prediction_tolerance=ALIGNMENT_TOLERANCE,
        identity_signature=affine_signature,
    )
    active_digest_before_routing = bank.digest()
    initial_queries = [
        (migration._new_representation, admitted_affine.slot_id),
        (drift._nonlinear_representation, admitted_nonlinear.slot_id),
    ]
    initial_routing_accuracy, initial_selected = _route_accuracy(
        bank, initial_queries, seed
    )
    shuffled_queries = [
        (migration._new_representation, admitted_nonlinear.slot_id),
        (drift._nonlinear_representation, admitted_affine.slot_id),
    ]
    shuffled_routing_accuracy, shuffled_selected = _route_accuracy(
        bank, shuffled_queries, seed
    )
    digest_after_routing = bank.digest()
    ambiguous_passive = _passive_source(migration._new_representation, seed + 99)
    ambiguous = bank.route_by_signature(
        torch.nn.functional.normalize(
            _signature(ambiguous_passive)
            + _signature(_passive_source(drift._nonlinear_representation, seed + 99)),
            dim=0,
        ),
        ambiguous_passive[:1],
    )
    missing = bank.route_by_signature(
        torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        ambiguous_passive[:1],
    )

    overflow = bank.admit_verified(
        "opaque-swapped",
        swapped,
        swapped_holdout_source,
        swapped_holdout_target,
        prediction_tolerance=ALIGNMENT_TOLERANCE,
        identity_signature=swapped_signature,
    )
    corrupted_receipt = bank.admit_verified(
        "opaque-corrupted",
        corrupted,
        corrupted_holdout_source,
        corrupted_holdout_target,
        prediction_tolerance=ALIGNMENT_TOLERANCE,
        identity_signature=corrupted_signature,
    )
    eviction = bank.evict_verified(
        admitted_affine.slot_id,
        lambda candidate: candidate.route_by_signature(
            _signature(_passive_source(drift._nonlinear_representation, seed + 10)),
            _passive_source(drift._nonlinear_representation, seed + 10)[:1],
        ).selected_slot_id
        == admitted_nonlinear.slot_id,
    )
    promoted = bank.promote_quarantined_verified(
        "opaque-swapped",
        swapped_holdout_source,
        swapped_holdout_target,
        prediction_tolerance=ALIGNMENT_TOLERANCE,
    )
    swapped_eval, swapped_route = _evaluate_routed(
        bank,
        model,
        evaluator,
        _swapped_representation,
        seed,
        query_expected_slot=promoted.slot_id,
    )
    corrupted_route = bank.route_by_signature(
        corrupted_signature,
        corrupted_holdout_source[:1],
    )
    restored = ExternalGoalRepresentationAlignmentBank.from_payload(bank.state_payload())
    restored_route = restored.route_by_signature(
        _signature(_passive_source(_swapped_representation, seed + 31)),
        swapped_holdout_source[:1],
    )

    gates = {
        "nonlinear_admitted_before_affine": admitted_nonlinear.accepted,
        "affine_admitted_after_nonlinear": admitted_affine.accepted,
        "initial_runtime_routing_accuracy": initial_routing_accuracy >= 1.0,
        "shuffled_identity_control_fails": shuffled_routing_accuracy < 0.5,
        "ambiguous_signature_refused": ambiguous.selected_slot_id is None,
        "missing_signature_refused": missing.selected_slot_id is None,
        "valid_overflow_quarantined": not overflow.accepted and overflow.quarantined,
        "corrupted_candidate_rejected_and_quarantined": (
            not corrupted_receipt.accepted and corrupted_receipt.quarantined
        ),
        "stable_eviction_accepted": eviction.accepted,
        "swapped_candidate_promoted": promoted.accepted,
        "promoted_runtime_routing": swapped_route.selected_slot_id == promoted.slot_id,
        "promoted_runtime_mastery": swapped_eval["mastery"] >= 0.95,
        "corrupted_candidate_not_served": (
            corrupted_route.selected_slot_id == admitted_nonlinear.slot_id
        ),
        "identity_routes_read_only": digest_after_routing == active_digest_before_routing,
        "exact_bank_persistence": restored.digest() == bank.digest(),
        "restored_runtime_route": restored_route.selected_slot_id == promoted.slot_id,
        "verifier_memory_unchanged": evaluator.digest() == evaluator_digest,
        "model_unchanged": model.digest() == model_digest,
        "controller_frozen": controller_digest == migration._digest(controller),
        "verifier_replay_zero": evaluator.sample_count.item() == outcome.shape[0],
        "alignment_replay_zero": all(
            adapter.sample_count.item() == nonlinear_source.shape[0] // TRAIN_STRIDE
            for adapter in (affine, nonlinear, swapped, corrupted)
        ),
    }
    report = {
        "schema": "neural-computer.external-goal-representation-alignment-identity.v1",
        "claim_boundary": (
            "bounded learned opaque signature routing for concurrent external goal "
            "alignments with ambiguity refusal; not semantic identity discovery, "
            "unrestricted growth, or general continual learning"
        ),
        "seed": seed,
        "configuration": {
            "capacity": BANK_CAPACITY,
            "quarantine_capacity": QUARANTINE_CAPACITY,
            "identity_width": IDENTITY_WIDTH,
            "identity_min_score": IDENTITY_MIN_SCORE,
            "identity_min_margin": IDENTITY_MIN_MARGIN,
            "feature_width": FEATURE_WIDTH,
            "runtime_frontend_ids_supplied": False,
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "metrics": {
            "initial_routing_accuracy": initial_routing_accuracy,
            "initial_selected_slots": initial_selected,
            "shuffled_routing_accuracy": shuffled_routing_accuracy,
            "shuffled_selected_slots": shuffled_selected,
            "ambiguous_route": _route_summary(ambiguous),
            "missing_route": _route_summary(missing),
            "overflow_admission": asdict(overflow),
            "corrupted_admission": asdict(corrupted_receipt),
            "eviction": asdict(eviction),
            "promoted_admission": asdict(promoted),
            "promoted_route": _route_summary(swapped_route),
            "promoted_mastery": swapped_eval,
            "corrupted_route": _route_summary(corrupted_route),
            "restored_route": _route_summary(restored_route),
            "final_active_slots": list(bank.slot_ids),
            "final_quarantine": list(bank.quarantined_space_ids),
        },
        "accounting": {
            "unique_verifier_outcomes": int(outcome.shape[0]),
            "unique_alignment_pairs": sum(
                int(adapter.sample_count.item())
                for adapter in (affine, nonlinear, swapped, corrupted)
            ),
            "heldout_alignment_pairs": sum(
                int(source.shape[0])
                for source in (
                    affine_source,
                    nonlinear_holdout_source,
                    swapped_holdout_source,
                    corrupted_holdout_source,
                )
            ),
            "unique_identity_signatures": 4,
            "transition_rows_consumed_once": transition_rows,
            "verifier_statistics_updates": 1,
            "alignment_statistics_updates": 4,
            "identity_route_updates_after_admission": 0,
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
    parser.add_argument("--seed", type=int, default=84801)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()

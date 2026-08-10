"""Promote multi-mask external identity-memory consolidation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from experiments.external_goal_alignment_drift_missing_reversal import train as drift
from experiments.external_goal_alignment_masked_capacity import train as capacity
from experiments.external_goal_representation_drift_gate import train as drift_data
from experiments.external_goal_representation_migration import train as migration
from experiments.external_one_pass_goal_evaluator import train as one_pass
from neural_computer import (
    AmodalCognitiveController,
    ExternalGoalEvaluatorStatistics,
    ExternalGoalRepresentationAlignmentBank,
)

INITIAL_PROTOTYPES_PER_SLOT = 1
GROWN_PROTOTYPES_PER_SLOT = 4
IDENTITY_MIN_SCORE = 0.55
IDENTITY_MIN_MARGIN = 0.04
OLD_MASK = torch.tensor(
    [False, True, True, False, True, True, True, True],
    dtype=torch.bool,
)
NEW_MASK = torch.tensor(
    [True, False, True, True, True, True, False, True],
    dtype=torch.bool,
)
THIRD_MASK = torch.tensor(
    [True, True, False, True, True, False, True, True],
    dtype=torch.bool,
)


def _route_summary(result) -> dict[str, object]:
    return {
        "selected_slot_id": result.selected_slot_id,
        "eligible_slot_ids": list(result.eligible_slot_ids),
        "scores": [float(value) for value in result.scores.tolist()],
        "margin": result.margin,
        "reason": result.reason,
    }


def _digest(module: torch.nn.Module) -> str:
    return migration._digest(module)


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

    model, transition_rows = migration.learned_goal._transition_model()
    model_digest = model.digest()
    old_state, old_goal, outcome = one_pass._verifier_batch(seed)
    evaluator = ExternalGoalEvaluatorStatistics(1, ridge=1e-5)
    evaluator.observe(old_state, old_goal, outcome)
    evaluator_digest = evaluator.digest()

    affine, affine_source, affine_target, affine_base = capacity.identity._fit_affine(seed)
    nonlinear_source, nonlinear_target = drift_data._nonlinear_batch(seed)
    nonlinear, nonlinear_holdout_source, nonlinear_holdout_target, nonlinear_base = (
        capacity.identity._fit_nonlinear(nonlinear_source, nonlinear_target, seed)
    )
    affine_base = torch.nn.functional.normalize(affine_base, dim=0)
    nonlinear_base = torch.nn.functional.normalize(nonlinear_base, dim=0)
    direction = drift._orthogonal_direction(affine_base, nonlinear_base, seed + 17)
    old_masked = drift._drifted_signature(affine_base, direction, 0.6)
    new_masked = drift._drifted_signature(affine_base, direction, -0.6)
    third_masked = affine_base

    bank = ExternalGoalRepresentationAlignmentBank(
        1,
        capacity=2,
        identity_width=8,
        identity_min_score=IDENTITY_MIN_SCORE,
        identity_min_margin=IDENTITY_MIN_MARGIN,
        identity_max_prototypes_per_slot=INITIAL_PROTOTYPES_PER_SLOT,
        identity_merge_cosine=0.94,
    )
    affine_receipt = bank.admit_verified(
        "opaque-affine",
        affine,
        affine_source,
        affine_target,
        prediction_tolerance=capacity.identity.ALIGNMENT_TOLERANCE,
        identity_signature=affine_base,
    )
    nonlinear_receipt = bank.admit_verified(
        "opaque-nonlinear",
        nonlinear,
        nonlinear_holdout_source,
        nonlinear_holdout_target,
        prediction_tolerance=capacity.identity.ALIGNMENT_TOLERANCE,
        identity_signature=nonlinear_base,
    )
    identity_memory = bank.identity_memory
    assert identity_memory is not None
    before_growth = identity_memory.digest()
    rejected_growth = identity_memory.grow_verified(
        GROWN_PROTOTYPES_PER_SLOT,
        lambda _candidate: False,
    )
    after_rejected_growth = identity_memory.digest()
    rejected_growth_preserved_state = (
        not rejected_growth.accepted
        and before_growth == after_rejected_growth
        and identity_memory.max_prototypes_per_slot == INITIAL_PROTOTYPES_PER_SLOT
    )

    def growth_retention_probe(candidate) -> bool:
        return all(
            candidate.propose(
                query,
                bank.slot_ids,
                minimum_score=IDENTITY_MIN_SCORE,
                query_mask=query_mask,
            ).selected_slot_id
            == expected_slot
            for query, query_mask, expected_slot in (
                (affine_base, None, affine_receipt.slot_id),
                (nonlinear_base, None, nonlinear_receipt.slot_id),
            )
        )

    accepted_growth = identity_memory.grow_verified(
        GROWN_PROTOTYPES_PER_SLOT,
        growth_retention_probe,
    )
    old_anchor = bank.accept_identity_anchor(
        old_masked,
        signature_mask=OLD_MASK,
        verifier_accepted=True,
    )
    new_anchor = bank.accept_identity_anchor(
        new_masked,
        signature_mask=NEW_MASK,
        verifier_accepted=True,
    )
    third_anchor = bank.accept_identity_anchor(
        third_masked,
        signature_mask=THIRD_MASK,
        verifier_accepted=True,
    )
    prototype_count_after_growth = identity_memory.prototype_count(
        affine_receipt.slot_id
    )
    before_consolidation = identity_memory.digest()
    rejected_consolidation = identity_memory.consolidate_verified(
        affine_receipt.slot_id,
        (2, 1),
        retention_probe=lambda _candidate: False,
    )
    after_rejected_consolidation = identity_memory.digest()
    rejected_consolidation_preserved_state = (
        not rejected_consolidation.accepted
        and before_consolidation == after_rejected_consolidation
        and identity_memory.prototype_count(affine_receipt.slot_id)
        == prototype_count_after_growth
    )

    def consolidation_retention_probe(candidate) -> bool:
        return (
            candidate.prototype_count(affine_receipt.slot_id) == 3
            and all(
                candidate.propose(
                    query,
                    bank.slot_ids,
                    minimum_score=IDENTITY_MIN_SCORE,
                    query_mask=query_mask,
                ).selected_slot_id
                == affine_receipt.slot_id
                for query, query_mask in (
                    (affine_base, None),
                    (old_masked, OLD_MASK),
                    (new_masked, NEW_MASK),
                    (third_masked, THIRD_MASK),
                )
            )
        )

    accepted_consolidation = identity_memory.consolidate_verified(
        affine_receipt.slot_id,
        (1, 2),
        consolidation_retention_probe,
    )
    retained_affine = bank.route_by_signature(affine_base, affine_source[:1])
    retained_nonlinear = bank.route_by_signature(
        nonlinear_base,
        nonlinear_source[:1],
    )
    retained_old = bank.route_by_signature(
        old_masked,
        affine_source[:1],
        signature_mask=OLD_MASK,
    )
    retained_new = bank.route_by_signature(
        new_masked,
        affine_source[:1],
        signature_mask=NEW_MASK,
    )
    retained_third = bank.route_by_signature(
        third_masked,
        affine_source[:1],
        signature_mask=THIRD_MASK,
    )
    affine_eval, affine_route = capacity.identity._evaluate_routed(
        bank,
        model,
        evaluator,
        migration._new_representation,
        seed + 501,
        query_expected_slot=affine_receipt.slot_id,
    )
    nonlinear_eval, nonlinear_route = capacity.identity._evaluate_routed(
        bank,
        model,
        evaluator,
        drift_data._nonlinear_representation,
        seed + 502,
        query_expected_slot=nonlinear_receipt.slot_id,
    )
    restored = ExternalGoalRepresentationAlignmentBank.from_payload(
        bank.state_payload()
    )
    restored_third = restored.route_by_signature(
        third_masked,
        affine_source[:1],
        signature_mask=THIRD_MASK,
    )

    gates = {
        "both_frontends_admitted": affine_receipt.accepted and nonlinear_receipt.accepted,
        "rejected_growth_preserves_state": rejected_growth_preserved_state,
        "accepted_growth_passes_retention": (
            accepted_growth.accepted
            and identity_memory.max_prototypes_per_slot == GROWN_PROTOTYPES_PER_SLOT
        ),
        "three_masked_patterns_appended": (
            old_anchor.anchor_update_stored
            and new_anchor.anchor_update_stored
            and third_anchor.anchor_update_stored
            and prototype_count_after_growth == 4
        ),
        "rejected_consolidation_preserves_state": rejected_consolidation_preserved_state,
        "accepted_consolidation_passes_retention": (
            accepted_consolidation.accepted
            and identity_memory.prototype_count(affine_receipt.slot_id) == 3
        ),
        "all_full_and_partial_routes_retained": all(
            route.selected_slot_id == affine_receipt.slot_id
            for route in (
                retained_affine,
                retained_old,
                retained_new,
                retained_third,
            )
        ),
        "nonlinear_route_retained": retained_nonlinear.selected_slot_id
        == nonlinear_receipt.slot_id,
        "restored_third_mask_route": restored_third.selected_slot_id
        == affine_receipt.slot_id,
        "affine_mastery_after_consolidation": affine_eval["mastery"] >= 0.95,
        "nonlinear_mastery_after_consolidation": nonlinear_eval["mastery"] >= 0.95,
        "exact_final_persistence": restored.digest() == bank.digest(),
        "verifier_memory_unchanged": evaluator.digest() == evaluator_digest,
        "model_unchanged": model.digest() == model_digest,
        "controller_frozen": controller_digest == _digest(controller),
        "verifier_replay_zero": evaluator.sample_count.item() == outcome.shape[0],
        "alignment_replay_zero": all(
            adapter.sample_count.item() == affine_source.shape[0]
            for adapter in (affine, nonlinear)
        ),
    }
    report = {
        "schema": "neural-computer.external-goal-representation-alignment-masked-consolidation.v1",
        "claim_boundary": (
            "bounded verifier-gated external identity-memory growth from one to four "
            "prototypes per slot, followed by retention-verified consolidation of "
            "two rows while retaining three differently masked patterns; not "
            "autonomous compression policy, unbounded growth, semantic open-world "
            "identity, or general continual learning"
        ),
        "seed": seed,
        "configuration": {
            "initial_max_prototypes_per_slot": INITIAL_PROTOTYPES_PER_SLOT,
            "grown_max_prototypes_per_slot": GROWN_PROTOTYPES_PER_SLOT,
            "identity_min_score": IDENTITY_MIN_SCORE,
            "identity_min_margin": IDENTITY_MIN_MARGIN,
            "mask_count": 3,
            "runtime_frontend_ids_used_for_updates": False,
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "metrics": {
            "rejected_growth": rejected_growth.__dict__,
            "accepted_growth": accepted_growth.__dict__,
            "rejected_consolidation": rejected_consolidation.__dict__,
            "accepted_consolidation": accepted_consolidation.__dict__,
            "retained_affine_route": _route_summary(retained_affine),
            "retained_nonlinear_route": _route_summary(retained_nonlinear),
            "retained_old_masked_route": _route_summary(retained_old),
            "retained_new_masked_route": _route_summary(retained_new),
            "retained_third_masked_route": _route_summary(retained_third),
            "restored_third_masked_route": _route_summary(restored_third),
            "affine_route": _route_summary(affine_route),
            "nonlinear_route": _route_summary(nonlinear_route),
            "affine_mastery": affine_eval,
            "nonlinear_mastery": nonlinear_eval,
            "prototype_counts": {
                str(slot_id): identity_memory.prototype_count(slot_id)
                for slot_id in identity_memory.slot_ids
            },
            "masked_prototype_count": identity_memory.masked_prototype_count,
            "restored_masked_prototype_count": restored.identity_memory.masked_prototype_count,
            "active_slots": list(bank.slot_ids),
        },
        "accounting": {
            "unique_verifier_outcomes": int(outcome.shape[0]),
            "unique_alignment_pairs": int(
                affine.sample_count.item() + nonlinear.sample_count.item()
            ),
            "heldout_alignment_pairs": int(
                affine_source.shape[0] + nonlinear_holdout_source.shape[0]
            ),
            "unique_masked_identity_queries": 3,
            "identity_memory_growth_operations": 2,
            "identity_memory_consolidation_operations": 2,
            "identity_anchor_operations": 3,
            "replayed_examples": 0,
            "transition_rows_consumed_once": transition_rows,
            "verifier_statistics_updates": 1,
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
    parser.add_argument("--seed", type=int, default=85301)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()

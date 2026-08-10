"""Promote verifier-gated growth of masked external identity memory."""

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
GROWN_PROTOTYPES_PER_SLOT = 3
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
        and identity_memory.max_prototypes_per_slot
        == INITIAL_PROTOTYPES_PER_SLOT
    )

    def growth_retention_probe(candidate) -> bool:
        affine_route = candidate.propose(
            affine_base,
            bank.slot_ids,
            minimum_score=IDENTITY_MIN_SCORE,
        )
        nonlinear_route = candidate.propose(
            nonlinear_base,
            bank.slot_ids,
            minimum_score=IDENTITY_MIN_SCORE,
        )
        return (
            candidate.max_prototypes_per_slot == GROWN_PROTOTYPES_PER_SLOT
            and affine_route.selected_slot_id == affine_receipt.slot_id
            and nonlinear_route.selected_slot_id == nonlinear_receipt.slot_id
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
    old_masked_pattern_appended = (
        old_anchor.accepted
        and old_anchor.anchor_update_stored
        and identity_memory.prototype_count(affine_receipt.slot_id) == 2
    )
    new_anchor = bank.accept_identity_anchor(
        new_masked,
        signature_mask=NEW_MASK,
        verifier_accepted=True,
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
    affine_eval, affine_route = capacity.identity._evaluate_routed(
        bank,
        model,
        evaluator,
        migration._new_representation,
        seed + 401,
        query_expected_slot=affine_receipt.slot_id,
    )
    nonlinear_eval, nonlinear_route = capacity.identity._evaluate_routed(
        bank,
        model,
        evaluator,
        drift_data._nonlinear_representation,
        seed + 402,
        query_expected_slot=nonlinear_receipt.slot_id,
    )
    restored = ExternalGoalRepresentationAlignmentBank.from_payload(
        bank.state_payload()
    )
    restored_new = restored.route_by_signature(
        new_masked,
        affine_source[:1],
        signature_mask=NEW_MASK,
    )

    gates = {
        "both_frontends_admitted": affine_receipt.accepted and nonlinear_receipt.accepted,
        "rejected_growth_preserves_state": rejected_growth_preserved_state,
        "accepted_growth_passes_retention": (
            accepted_growth.accepted
            and identity_memory.max_prototypes_per_slot == GROWN_PROTOTYPES_PER_SLOT
        ),
        "old_full_routes_retained": (
            retained_affine.selected_slot_id == affine_receipt.slot_id
            and retained_nonlinear.selected_slot_id == nonlinear_receipt.slot_id
        ),
        "old_masked_pattern_appended": (
            old_masked_pattern_appended
        ),
        "new_masked_pattern_appended_without_replacement": (
            new_anchor.accepted
            and new_anchor.anchor_update_stored
            and identity_memory.prototype_count(affine_receipt.slot_id) == 3
        ),
        "both_partial_patterns_route": (
            retained_old.selected_slot_id == affine_receipt.slot_id
            and retained_new.selected_slot_id == affine_receipt.slot_id
        ),
        "restored_partial_pattern_routes": (
            restored_new.selected_slot_id == affine_receipt.slot_id
        ),
        "affine_mastery_after_growth": affine_eval["mastery"] >= 0.95,
        "nonlinear_mastery_after_growth": nonlinear_eval["mastery"] >= 0.95,
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
        "schema": "neural-computer.external-goal-representation-alignment-masked-growth.v1",
        "claim_boundary": (
            "bounded verifier-gated external identity-memory capacity growth from one "
            "to three prototypes per slot, followed by two distinct masked-pattern "
            "insertions; not autonomous retention policy, unbounded growth, semantic "
            "open-world identity, or general continual learning"
        ),
        "seed": seed,
        "configuration": {
            "initial_max_prototypes_per_slot": INITIAL_PROTOTYPES_PER_SLOT,
            "grown_max_prototypes_per_slot": GROWN_PROTOTYPES_PER_SLOT,
            "identity_min_score": IDENTITY_MIN_SCORE,
            "identity_min_margin": IDENTITY_MIN_MARGIN,
            "new_mask_matches_old_mask": bool(torch.equal(NEW_MASK, OLD_MASK)),
            "runtime_frontend_ids_used_for_updates": False,
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "metrics": {
            "rejected_growth": rejected_growth.__dict__,
            "accepted_growth": accepted_growth.__dict__,
            "old_masked_anchor": old_anchor.__dict__,
            "new_masked_anchor": new_anchor.__dict__,
            "retained_affine_route": _route_summary(retained_affine),
            "retained_nonlinear_route": _route_summary(retained_nonlinear),
            "retained_old_masked_route": _route_summary(retained_old),
            "retained_new_masked_route": _route_summary(retained_new),
            "restored_new_masked_route": _route_summary(restored_new),
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
            "unique_masked_identity_queries": 2,
            "identity_memory_growth_operations": 2,
            "identity_anchor_operations": 2,
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
    parser.add_argument("--seed", type=int, default=85201)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
